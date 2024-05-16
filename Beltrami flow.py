import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyDOE import lhs
import time
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import scipy.io

data = [[] for i in range(4)]
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, X_ub, X_f, layers):
        self.lb = lb
        self.ub = ub

        self.losssum = []
        self.lossres = []
        self.lossbc = []
        self.lossin = []
        self.it = 0

        self.x_u = X_u[:, 0:1]  # x点的坐标
        self.y_u = X_u[:, 1:2]  # y点的坐标
        self.z_u = X_u[:, 2:3]
        self.t_u = X_u[:, 3:4]  # t点坐标

        self.x_ub = X_ub[:, 0:1]  # x点的坐标
        self.y_ub = X_ub[:, 1:2]  # y点的坐标
        self.z_ub = X_ub[:, 2:3]
        self.t_ub = X_ub[:, 3:4]  # t 点的坐标

        self.x_f = X_f[:, 0:1]  # x点的坐标
        self.y_f = X_f[:, 1:2]  # y点的坐标
        self.z_f = X_f[:, 2:3]
        self.t_f = X_f[:, 3:4]

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        self.encoder_weights_1 = self.xavier_init([4, layers[1]])
        self.encoder_biases_1 = self.xavier_init([1, layers[1]])

        self.encoder_weights_2 = self.xavier_init([4, layers[1]])
        self.encoder_biases_2 = self.xavier_init([1, layers[1]])

        self.encoder_weights_3 = self.xavier_init([4, layers[1]])
        self.encoder_biases_3 = self.xavier_init([1, layers[1]])



        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.sigma1 = tf.Variable([0.0], dtype=tf.float32)
        self.sigma2 = tf.Variable([0.0], dtype=tf.float32)
        self.sigma3 = tf.Variable([0.0], dtype=tf.float32)
        self.a = tf.constant([0.5], dtype=tf.float32)

        # 此时可以调用GPUwith tf.Session() as sess,tf.device('/gpu:0'):
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, self.y_u.shape[1]])
        self.z_u_tf = tf.placeholder(tf.float32, shape=[None, self.z_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.y_ub_tf = tf.placeholder(tf.float32, shape=[None, self.y_ub.shape[1]])
        self.z_ub_tf = tf.placeholder(tf.float32, shape=[None, self.z_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None, self.z_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred, self.v_pred, self.w_pred, self.p_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.z_u_tf,
                                                                        self.t_u_tf)
        self.ini_u, self.ini_v, self.ini_w, self.ini_p = self.init_u(self.x_u_tf, self.y_u_tf, self.z_u_tf, self.t_u_tf)

        self.ub_pred, self.vb_pred, self.wb_pred, self.pb_pred = self.net_u(self.x_ub_tf, self.y_ub_tf, self.z_ub_tf,
                                                                            self.t_ub_tf)
        self.ub_exact, self.vb_exact, self.wb_exact, self.pb_exact = self.bound(self.x_ub_tf, self.y_ub_tf,
                                                                                self.z_ub_tf, self.t_ub_tf)
        self.f1, self.f2, self.f3, self.f4 = self.net_f(self.x_f_tf, self.y_f_tf, self.z_f_tf, self.t_f_tf)

        self.sigma1 = tf.Variable([0.0], dtype=tf.float32)
        self.sigma2 = tf.Variable([0.0], dtype=tf.float32)
        self.sigma3 = tf.Variable([0.0], dtype=tf.float32)

        self.loss_res = tf.reduce_mean(tf.square(self.f1)) + \
                        tf.reduce_mean(tf.square(self.f2)) + \
                        tf.reduce_mean(tf.square(self.f3)) + \
                        tf.reduce_mean(tf.square(self.f4))

        self.loss_in = tf.reduce_mean(tf.square(self.u_pred - self.ini_u)) + \
                       tf.reduce_mean(tf.square(self.v_pred - self.ini_v)) + \
                       tf.reduce_mean(tf.square(self.w_pred - self.ini_w)) + \
                       tf.reduce_mean(tf.square(self.p_pred - self.ini_p))

        self.loss_bc = tf.reduce_mean(tf.square(self.ub_pred - self.ub_exact)) + \
                       tf.reduce_mean(tf.square(self.vb_pred - self.vb_exact)) + \
                       tf.reduce_mean(tf.square(self.wb_pred - self.wb_exact)) + \
                       tf.reduce_mean(tf.square(self.pb_pred - self.pb_exact))

        self.loss1 = self.likelihood_loss()
        self.loss = self.true_loss()

        self.loss = self.loss_bc + self.loss_in + self.loss_res

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 5000000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def likelihood_loss(self):
        loss1 = tf.stop_gradient(self.loss_bc)
        loss2 = tf.stop_gradient(self.loss_in)
        loss3 = tf.stop_gradient(self.loss_res)
        loss = self.a*tf.exp(-self.sigma1) * loss1 + self.sigma1 \
               + self.a*tf.exp(-self.sigma2) * loss2 + self.sigma2 \
               + self.a*tf.exp(-self.sigma3) * loss3 + self.sigma3
        return loss

    def true_loss(self):
        w1 = tf.stop_gradient(self.sigma1)
        w2 = tf.stop_gradient(self.sigma2)
        w3 = tf.stop_gradient(self.sigma3)
        return self.a*tf.exp(-w1) * self.loss_bc + self.sigma1\
                +self.a*tf.exp(-w2) * self.loss_in + self.sigma2\
                +self.a*tf.exp(-w3) * self.loss_res+ self.sigma3

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    """def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        return tf.Variable(tf.random_normal([in_dim, out_dim]), dtype=tf.float32)"""

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        #xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        #return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,dtype=tf.float32)


    def neural_net(self, H):
        num_layers = len(self.layers)
        encoder_1 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_1), self.encoder_biases_1))
        encoder_2 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_2), self.encoder_biases_2))
        encoder_3 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_3), self.encoder_biases_3))
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + \
                tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + \
                tf.math.multiply(1 - tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_2- tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_3)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H


    def init_u(self, x, y, z, t):
        a = d = 1
        u = -a * (tf.exp(a * x) * tf.sin(a * y + d * z) + tf.exp(a * z) * tf.cos(a * x + d * y)) * tf.exp(-d ** 2 * t)
        v = -a * (tf.exp(a * y) * tf.sin(a * z + d * x) + tf.exp(a * x) * tf.cos(a * y + d * z)) * tf.exp(-d ** 2 * t)
        w = -a * (tf.exp(a * z) * tf.sin(a * x + d * y) + tf.exp(a * y) * tf.cos(a * z + d * x)) * tf.exp(-d ** 2 * t)
        p = (-1 / 2) * a ** 2 * (
                    tf.exp(2 * a * x) + tf.exp(2 * a * y) + tf.exp(2 * a * z) + 2 * tf.sin(a * x + d * y) * tf.cos(
                a * z + d * x) * tf.exp(a * (y + z)) +
                    2 * tf.sin(a * y + d * z) * tf.cos(a * x + d * y) * tf.exp(a * (z + x)) + 2 * tf.sin(
                a * z + d * x) * tf.cos(a * y + d * z) * tf.exp(a * (x + y))) * tf.exp(-2 * d ** 2 * t)
        return u, v, w, p

    def net_u(self, x, y, z, t):
        u_w_p = self.neural_net(tf.concat([x, y, z, t], 1))
        u = u_w_p[:, 0:1]
        v = u_w_p[:, 1:2]
        w = u_w_p[:, 2:3]
        p = u_w_p[:, 3:4]
        return u, v, w, p

    def bound(self, x, y, z, t):
        a = d = 1
        u = -a * (tf.exp(a * x) * tf.sin(a * y + d * z) + tf.exp(a * z) * tf.cos(a * x + d * y)) * tf.exp(-d ** 2 * t)
        v = -a * (tf.exp(a * y) * tf.sin(a * z + d * x) + tf.exp(a * x) * tf.cos(a * y + d * z)) * tf.exp(-d ** 2 * t)
        w = -a * (tf.exp(a * z) * tf.sin(a * x + d * y) + tf.exp(a * y) * tf.cos(a * z + d * x)) * tf.exp(-d ** 2 * t)
        p = (-1 / 2) * a ** 2 * (
                tf.exp(2 * a * x) + tf.exp(2 * a * y) + tf.exp(2 * a * z) + 2 * tf.sin(a * x + d * y) * tf.cos(
            a * z + d * x) * tf.exp(a * (y + z)) +
                2 * tf.sin(a * y + d * z) * tf.cos(a * x + d * y) * tf.exp(a * (z + x)) + 2 * tf.sin(
            a * z + d * x) * tf.cos(a * y + d * z) * tf.exp(a * (x + y))) * tf.exp(-2 * d ** 2 * t)
        return u, v, w, p

    def you(self, x, y, z, t):
        RE=200
        u,v,w,p = self.bound(x, y, z, t)
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        w_t = tf.gradients(w, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]
        f1 = u_t + (-1 / RE) * (u_xx + u_yy + u_zz) + u * u_x + v * u_y + w * u_z + p_x
        f2 = v_t + (-1 / RE) * (v_xx + v_yy + v_zz) + u * v_x + v * v_y + w * v_z + p_y
        f3 = w_t + (-1 / RE) * (w_xx + w_yy + w_zz) + u * w_x + v * w_y + w * w_z + p_z
        f4 = u_x + v_y + w_z
        return f1, f2, f3, f4

    def net_f(self, x, y, z,t):
        n = 200
        Re = 10
        u, v, w, p = self.net_u(x, y, z,t)
        dx = (1 - 0) / (n - 1)  # / 2
        dy = (1 - 0) / (n - 1)  # / 2
        dz = (1 - 0) / (n - 1)

        xE, xW = x + dx, x - dx
        yN, yS = y + dy, y - dy
        zU, zD = z + dz, z - dz

        uE, vE, wE, pE = self.net_u(xE, y, z, t)
        uW, vW, wW, pW = self.net_u(xW, y, z, t)
        uN, vN, wN, pN = self.net_u(x, yN, z, t)
        uS, vS, wS, pS = self.net_u(x, yS, z, t)
        uU, vU, wU, pU = self.net_u(x, y, zU, t)
        uD, vD, wD, pD = self.net_u(x, y, zD, t)

        uE_x = tf.gradients(uE, xE)[0]
        uW_x = tf.gradients(uW, xW)[0]
        uN_y = tf.gradients(uN, yN)[0]
        uS_y = tf.gradients(uS, yS)[0]
        uU_z = tf.gradients(uU, zU)[0]
        uD_z = tf.gradients(uD, zD)[0]

        vE_x = tf.gradients(vE, xE)[0]
        vW_x = tf.gradients(vW, xW)[0]
        vN_y = tf.gradients(vN, yN)[0]
        vS_y = tf.gradients(vS, yS)[0]
        vU_z = tf.gradients(vU, zU)[0]
        vD_z = tf.gradients(vD, zD)[0]

        wE_x = tf.gradients(wE, xE)[0]
        wW_x = tf.gradients(wW, xW)[0]
        wN_y = tf.gradients(wN, yN)[0]
        wS_y = tf.gradients(wS, yS)[0]
        wU_z = tf.gradients(wU, zU)[0]
        wD_z = tf.gradients(wD, zD)[0]

        pE_x = tf.gradients(pE, xE)[0]
        pW_x = tf.gradients(pW, xW)[0]
        pN_y = tf.gradients(pN, yN)[0]
        pS_y = tf.gradients(pS, yS)[0]
        pU_z = tf.gradients(pU, zU)[0]
        pD_z = tf.gradients(pD, zD)[0]

        # can 1nd upwind
        uc_e, uc_w = 0.5 * (uE + u), 0.5 * (uW + u)
        vc_n, vc_s = 0.5 * (vN + v), 0.5 * (vS + v)
        wc_u, wc_d = 0.5 * (wU + u), 0.5 * (wD + u)
        div = (uc_e - uc_w) / dx + (vc_n - vc_s) / dy + (wc_u - wc_d) / dz

        # can 2nd upwind
        u_x = tf.gradients(u, x)[0]
        Uem_cuw2 = u + u_x * dx / 2.0  # + (uE_x - u_x)*dx /8.0
        Uep_cuw2 = uE - uE_x * dx / 2.0  # + (uE_x - u_x)*dx /8.0
        Uwm_cuw2 = uW + uW_x * dx / 2.0  # + (u_x - uW_x)*dx /8.0
        Uwp_cuw2 = u - u_x * dx / 2.0  # + (u_x - uW_x)*dx /8.0
        Ue_cuw2 = tf.where(tf.greater_equal(uc_e, 0.0), Uem_cuw2, Uep_cuw2)
        Uw_cuw2 = tf.where(tf.greater_equal(uc_w, 0.0), Uwm_cuw2, Uwp_cuw2)

        u_y = tf.gradients(u, y)[0]
        Unm_cuw2 = u + u_y * dy / 2.0  # + (uN_y - u_y)*dy /8.0
        Unp_cuw2 = uN - uN_y * dy / 2.0  # + (uN_y - u_y)*dy /8.0
        Usm_cuw2 = uS + uS_y * dy / 2.0  # + (u_y - uS_y)*dy /8.0
        Usp_cuw2 = u - u_y * dy / 2.0  # + (u_y - uS_y)*dy /8.0
        Un_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Unm_cuw2, Unp_cuw2)
        Us_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Usm_cuw2, Usp_cuw2)

        u_z = tf.gradients(u, z)[0]
        Uum_cuw2 = u + u_z * dz / 2.0  # + (uN_y - u_y)*dy /8.0
        Uup_cuw2 = uU - uU_z * dz / 2.0  # + (uN_y - u_y)*dy /8.0
        Udm_cuw2 = uD + uD_z * dz / 2.0  # + (u_y - uS_y)*dy /8.0
        Udp_cuw2 = u - u_z * dz / 2.0  # + (u_y - uS_y)*dy /8.0
        Uu_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Uum_cuw2, Uup_cuw2)
        Ud_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Udm_cuw2, Udp_cuw2)

        v_x = tf.gradients(v, x)[0]
        Vem_cuw2 = v + v_x * dx / 2.0  # + (vE_x - v_x)*dx /8.0
        Vep_cuw2 = vE - vE_x * dx / 2.0  # + (vE_x - v_x)*dx /8.0
        Vwm_cuw2 = vW + vW_x * dx / 2.0  # + (v_x - vW_x)*dx /8.0
        Vwp_cuw2 = v - v_x * dx / 2.0  # + (v_x - vW_x)*dx /8.0
        Ve_cuw2 = tf.where(tf.greater_equal(uc_e, 0.0), Vem_cuw2, Vep_cuw2)
        Vw_cuw2 = tf.where(tf.greater_equal(uc_w, 0.0), Vwm_cuw2, Vwp_cuw2)

        v_y = tf.gradients(v, y)[0]
        Vnm_cuw2 = v + v_y * dy / 2.0  # + (vN_y - v_y)*dy /8.0
        Vnp_cuw2 = vN - vN_y * dy / 2.0  # + (vN_y - v_y)*dy /8.0
        Vsm_cuw2 = vS + vS_y * dy / 2.0  # + (v_y - vS_y)*dy /8.0
        Vsp_cuw2 = v - v_y * dy / 2.0  # + (v_y - vS_y)*dy /8.0
        Vn_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Vnm_cuw2, Vnp_cuw2)
        Vs_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Vsm_cuw2, Vsp_cuw2)

        v_z = tf.gradients(v, z)[0]
        Vum_cuw2 = v + v_z * dz / 2.0  # + (vN_y - v_y)*dy /8.0
        Vup_cuw2 = vN - vU_z * dz / 2.0  # + (vN_y - v_y)*dy /8.0
        Vdm_cuw2 = vS + vD_z * dz / 2.0  # + (v_y - vS_y)*dy /8.0
        Vdp_cuw2 = v - v_z * dz / 2.0  # + (v_y - vS_y)*dy /8.0
        Vu_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Vum_cuw2, Vup_cuw2)
        Vd_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Vdm_cuw2, Vdp_cuw2)

        w_x = tf.gradients(w, x)[0]
        Wem_cuw2 = w + w_x * dx / 2.0  # + (vE_x - v_x)*dx /8.0
        Wep_cuw2 = wE - wE_x * dx / 2.0  # + (vE_x - v_x)*dx /8.0
        Wwm_cuw2 = wW + wW_x * dx / 2.0  # + (v_x - vW_x)*dx /8.0
        Wwp_cuw2 = w - w_x * dx / 2.0  # + (v_x - vW_x)*dx /8.0
        We_cuw2 = tf.where(tf.greater_equal(uc_e, 0.0), Wem_cuw2, Wep_cuw2)
        Ww_cuw2 = tf.where(tf.greater_equal(uc_w, 0.0), Wwm_cuw2, Wwp_cuw2)

        w_y = tf.gradients(w, y)[0]
        Wnm_cuw2 = w + w_y * dy / 2.0  # + (vN_y - v_y)*dy /8.0
        Wnp_cuw2 = wN - wN_y * dy / 2.0  # + (vN_y - v_y)*dy /8.0
        Wsm_cuw2 = wS + wS_y * dy / 2.0  # + (v_y - vS_y)*dy /8.0
        Wsp_cuw2 = w - w_y * dy / 2.0  # + (v_y - vS_y)*dy /8.0
        Wn_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Wnm_cuw2, Wnp_cuw2)
        Ws_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Wsm_cuw2, Wsp_cuw2)

        w_z = tf.gradients(w, z)[0]
        Wum_cuw2 = w + w_z * dz / 2.0  # + (vN_y - v_y)*dy /8.0
        Wup_cuw2 = wN - wU_z * dz / 2.0  # + (vN_y - v_y)*dy /8.0
        Wdm_cuw2 = wS + wD_z * dz / 2.0  # + (v_y - vS_y)*dy /8.0
        Wdp_cuw2 = w - w_z * dz / 2.0  # + (v_y - vS_y)*dy /8.0
        Wu_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Wum_cuw2, Wup_cuw2)
        Wd_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Wdm_cuw2, Wdp_cuw2)

        UUx_cuw2 = (uc_e * Ue_cuw2 - uc_w * Uw_cuw2) / dx
        VUy_cuw2 = (vc_n * Un_cuw2 - vc_s * Us_cuw2) / dy
        WUz_cuw2 = (wc_u * Uu_cuw2 - wc_d * Ud_cuw2) / dz

        UVx_cuw2 = (uc_e * Ve_cuw2 - uc_w * Vw_cuw2) / dx
        VVy_cuw2 = (vc_n * Vn_cuw2 - vc_s * Vs_cuw2) / dy
        WVz_cuw2 = (wc_u * Vu_cuw2 - wc_d * Vd_cuw2) / dz

        UWx_cuw2 = (uc_e * We_cuw2 - uc_w * Ww_cuw2) / dx
        VWy_cuw2 = (vc_n * Wn_cuw2 - vc_s * Ws_cuw2) / dy
        WWz_cuw2 = (wc_u * Wu_cuw2 - wc_d * Wd_cuw2) / dz

        # can 2nd central difference
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]
        pe_ccd2 = (p + pE) / 2.0 - (pE_x - p_x) * dx / 8.0
        pw_ccd2 = (pW + p) / 2.0 - (p_x - pW_x) * dx / 8.0
        pn_ccd2 = (p + pN) / 2.0 - (pN_y - p_y) * dy / 8.0
        ps_ccd2 = (pS + p) / 2.0 - (p_y - pS_y) * dy / 8.0
        pu_ccd2 = (p + pU) / 2.0 - (pU_z - p_z) * dz / 8.0
        pd_ccd2 = (pD + p) / 2.0 - (p_z - pD_z) * dz / 8.0

        Px_ccd2 = (pe_ccd2 - pw_ccd2) / dx
        Py_ccd2 = (pn_ccd2 - ps_ccd2) / dy
        Pz_ccd2 = (pu_ccd2 - pd_ccd2) / dz

        # 2nd central difference
        Uxx_cd2 = (uE - 2.0 * u + uW) / (dx * dx)
        Uyy_cd2 = (uN - 2.0 * u + uS) / (dy * dy)
        Uzz_cd2 = (uU - 2.0 * u + uD) / (dz * dz)

        Vxx_cd2 = (vE - 2.0 * v + vW) / (dx * dx)
        Vyy_cd2 = (vN - 2.0 * v + vS) / (dy * dy)
        Vzz_cd2 = (vU - 2.0 * v + vD) / (dz * dz)

        Wxx_cd2 = (wE - 2.0 * w + wW) / (dx * dx)
        Wyy_cd2 = (wN - 2.0 * w + wS) / (dy * dy)
        Wzz_cd2 = (wU - 2.0 * w + wD) / (dz * dz)

        f4 = div
        f1 = UUx_cuw2 + VUy_cuw2 + WUz_cuw2 - 1.0 / Re * (Uxx_cd2 + Uyy_cd2 + Uzz_cd2) - u * div + Px_ccd2
        f2 = UVx_cuw2 + VVy_cuw2 + WVz_cuw2 - 1.0 / Re * (Vxx_cd2 + Vyy_cd2 + Vzz_cd2) - v * div + Py_ccd2
        f3 = UWx_cuw2 + VWy_cuw2 + WWz_cuw2 - 1.0 / Re * (Wxx_cd2 + Wyy_cd2 + Wzz_cd2) - w * div + Pz_ccd2
        return f1, f2, f3, f4

    """def net_f(self, x, y, z, t):
        RE = 200
        u, v, w, p = self.net_u(x, y, z, t)
        f11, f22, f33, f44 = self.you(x, y, z, t)
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        w_t = tf.gradients(w, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]
        f1 = u_t + (-1 / RE) * (u_xx + u_yy + u_zz) + u * u_x + v * u_y + w * u_z + p_x - f11
        f2 = v_t + (-1 / RE) * (v_xx + v_yy + v_zz) + u * v_x + v * v_y + w * v_z + p_y - f22
        f3 = w_t + (-1 / RE) * (w_xx + w_yy + w_zz) + u * w_x + v * w_y + w * w_z + p_z - f33
        f4 = u_x + v_y + w_z
        return f1, f2, f3, f4"""

    def callback(self, loss, loss_bc, loss_in, loss_res, sigma1, sigma2, sigma3):

        if loss < 0.1:
            data[0].append(self.it)
            data[1].append(loss)
            # data[2].append(lambda_1)
            # data[3].append(np.exp(lambda_2))
            print(
                'It: %d,  Loss: %e, loss_bc:%.5e, loss_in:%.5e, loss_res:%.5e, sigma1:%.5f, sigma2:%.5f, sigma3:%.5f' %
                (self.it, loss, loss_bc, loss_in, loss_res, sigma1, sigma2, sigma3))

            self.losssum.append(loss)
            self.lossbc.append(loss_bc)
            self.lossin.append(loss_in)
            self.lossres.append(loss_res)
        self.it = self.it + 1

    def train(self, nIter):
        tf_dict = {self.x_u_tf: self.x_u, self.y_u_tf: self.y_u, self.z_u_tf: self.z_u, self.t_u_tf: self.t_u,
                   self.x_ub_tf: self.x_ub, self.y_ub_tf: self.y_ub, self.z_ub_tf: self.z_ub,
                   self.t_ub_tf: self.t_ub, self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.z_f_tf: self.z_f,
                   self.t_f_tf: self.t_f}

        start_time = time.time()
        for its in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            self.sess.run(self.train_op_Adam1, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            loss1_value = self.sess.run(self.loss1, tf_dict)
            w1 = self.sess.run(self.sigma1)
            w2 = self.sess.run(self.sigma2)
            w3 = self.sess.run(self.sigma3)
            if (loss_value < 1):
                print('It: %d, Loss: %.3e, Loss1: %.3e, sigma1:%.5f, sigma2:%.5f, sigma3:%.5f' %
                      (its, loss_value, loss1_value, w1, w2, w3))
                data[0].append(its)
                data[1].append(loss_value)
            self.it = self.it + 1
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_bc, self.loss_in, self.loss_res, self.sigma1,
                                             self.sigma2, self.sigma3],
                                    loss_callback=self.callback)
        elapsed = time.time() - start_time
        print('total time:%.2f' % elapsed)

    def predict(self, X, YZ, t):
        tf_dict = {self.x_u_tf: X, self.y_u_tf: YZ[:, 0:1], self.z_u_tf: YZ[:, 1:2], self.t_u_tf: t}
        U = self.sess.run(self.u_pred, tf_dict)
        P = self.sess.run(self.p_pred, tf_dict)
        V = self.sess.run(self.v_pred, tf_dict)
        W = self.sess.run(self.w_pred, tf_dict)
        return U, V, W, P


if __name__ == "__main__":
    N_u = 5766
    N_b = 5766
    N_f = 10000
    layers = [4, 50, 50, 50, 50, 50, 4]

    x = np.linspace(-1, 1, 40)
    y = np.linspace(-1, 1, 40)
    z = np.linspace(-1, 1, 40)
    t = np.linspace(0, 1, 41)
    X, Y, Z, T = np.meshgrid(x, y, z, t)

    init = np.hstack((np.array([X[:, :, :, 0:1].T.flatten().T]).T,  # 下 初始  t=0
                      np.array([Y[:, :, :, 0:1].T.flatten().T]).T,
                      np.array([Z[:, :, :, 0:1].T.flatten().T]).T,
                      np.array([T[:, :, :, 0:1].T.flatten().T]).T))

    bc1 = np.hstack((np.array([X[:, 0:1, :, :].T.flatten().T]).T,  # 左 x=-1
                     np.array([Y[:, 0:1, :, :].T.flatten().T]).T,
                     np.array([Z[:, 0:1, :, :].T.flatten().T]).T,
                     np.array([T[:, 0:1, :, :].T.flatten().T]).T))

    bc2 = np.hstack((np.array([X[:, -1, :, :].T.flatten().T]).T,  # 右 x=1
                     np.array([Y[:, :, 0:1, :].T.flatten().T]).T,
                     np.array([Z[:, :, 0:1, :].T.flatten().T]).T,
                     np.array([T[:, :, 0:1, :].T.flatten().T]).T))

    bc3 = np.hstack((np.array([X[0:1, :, :].T.flatten().T]).T,  # 前 y=-1
                     np.array([Y[0:1, :, :].flatten().T]).T,
                     np.array([Z[0:1, :, :].T.flatten().T]).T,
                     np.array([T[0:1, :, :].T.flatten().T]).T))

    bc4 = np.hstack((np.array([X[0:1, :, :].T.flatten().T]).T,  # 后 y=1
                     np.array([Y[-1, :, :].T.flatten().T]).T,
                     np.array([Z[0:1, :, :].T.flatten().T]).T,
                     np.array([T[0:1, :, :].T.flatten().T]).T))

    bc5 = np.hstack((np.array([X[:, :, 0:1, :].T.flatten().T]).T,  # z=-1
                     np.array([Y[:, :, 0:1, :].T.flatten().T]).T,
                     np.array([Z[:, :, 0:1, :].T.flatten().T]).T,
                     np.array([T[:, :, 0:1, :].T.flatten().T]).T))

    bc6 = np.hstack((np.array([X[:, :, 0:1, :].T.flatten().T]).T,  # z=1
                     np.array([Y[:, :, 0:1, :].T.flatten().T]).T,
                     np.array([Z[:, :, -1, :].T.flatten().T]).T,
                     np.array([T[:, :, 0:1, :].T.flatten().T]).T))

    bc = np.vstack([bc1, bc2, bc3, bc4, bc5, bc6])

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None], T.flatten()[:, None]))

    idx1 = np.random.choice(init.shape[0], N_u, replace=False)
    X_u_train = init[idx1, :]

    idx2 = np.random.choice(bc.shape[0], N_b, replace=False)
    X_ub_train = bc[idx2, :]
    # i = 0.1
    lb = X_star.min(0)
    ub = X_star.max(0)
    X_f_train = lb + (ub - lb) * lhs(4, N_f)
    model = PhysicsInformedNN(X_u_train, X_ub_train, X_f_train, layers)
    start_time = time.time()
    model.train(10000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    # y2 = np.ones([100, 1])*0.1
    x1 = np.ones([40, 1]) * 0.1
    X, Y2 = np.meshgrid(x1, x1)
    Y, Z = np.meshgrid(y, z)
    YZ = np.hstack((Y.flatten()[:, None], Z.flatten()[:, None]))
    for i in t:
        t1 = np.ones([YZ.shape[0], 1]) * i
        X1 = np.ones([YZ.shape[0], 1]) * 0.1


        def exact(x, y, z, t):
            a = d = 1
            u = -a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(
                -d ** 2 * t)
            v = -a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(
                -d ** 2 * t)
            w = -a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(
                -d ** 2 * t)
            p = (-1 / 2) * a ** 2 * (
                    np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) + 2 * np.sin(a * x + d * y) * np.cos(
                a * z + d * x) * np.exp(a * (y + z)) + 2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(
                a * (z + x)) + 2 * np.sin(
                a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(-2 * d ** 2 * t)
            return u, v, w, p


        u_exact, v_exact, w_exact, p_exact = exact(X, Y, Z, i)
        u1, v1, w1, p1 = model.predict(X1, YZ, t1)

        u_pred = griddata(YZ, u1.flatten(), (Y, Z), method='cubic')
        v_pred = griddata(YZ, v1.flatten(), (Y, Z), method='cubic')
        w_pred = griddata(YZ, w1.flatten(), (Y, Z), method='cubic')
        p_pred = griddata(YZ, p1.flatten(), (Y, Z), method='cubic')
        error_u = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
        error_v = np.linalg.norm(v_pred - v_exact, 2) / np.linalg.norm(v_exact, 2)
        error_w = np.linalg.norm(w_pred - w_exact, 2) / np.linalg.norm(w_exact, 2)
        error_p = np.linalg.norm(p_pred - p_exact, 2) / np.linalg.norm(p_exact, 2)
        Error_u = np.abs(u_pred - u_exact)
        Error_v = np.abs(v_pred - v_exact)
        Error_w = np.abs(w_pred - w_exact)
        Error_p = np.abs(p_pred - p_exact)

        """scipy.io.savemat('up.mat', {'a': u_pred})
        scipy.io.savemat('ue.mat', {'a1': u_exact})
        scipy.io.savemat('Eu.mat', {'a11': Error_u})

        scipy.io.savemat('vp.mat', {'b': v_pred})
        scipy.io.savemat('ve.mat', {'b1': v_exact})
        scipy.io.savemat('Ev.mat', {'b11': Error_v})

        scipy.io.savemat('wp.mat', {'d': w_pred})
        scipy.io.savemat('we.mat', {'d1': w_exact})
        scipy.io.savemat('Ew.mat', {'d11': Error_w})

        scipy.io.savemat('pp.mat', {'c': p_pred})
        scipy.io.savemat('pe.mat', {'c1': p_exact})
        scipy.io.savemat('Ep.mat', {'c11': Error_p})"""


        if i == 0.1:
            print(
                'time: %.2f ,Error u: %e,Error v: %e,Error w: %e,Error p: %e' % (i, error_u, error_v, error_w, error_p))
            fig1 = plt.figure(figsize=[18, 5])
            ####### Row 0: u(t,x) ##################
            plt.subplot(131)
            plt.colorbar(plt.imshow(u_exact, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('u_exact')
            plt.subplot(132)
            plt.colorbar(plt.imshow(u_pred, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('u_pred')
            plt.subplot(133)
            plt.colorbar(plt.imshow(np.abs(u_pred - u_exact), interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('u_error')
            plt.tight_layout()
            fig2 = plt.figure(figsize=[18, 5])
            ####### Row 0: u(t,x) ##################
            plt.subplot(131)
            plt.colorbar(plt.imshow(v_exact, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('v_exact')
            plt.subplot(132)
            plt.colorbar(plt.imshow(v_pred, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('v_pred')
            plt.subplot(133)
            plt.colorbar(plt.imshow(np.abs(v_pred - v_exact), interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('v_error')
            plt.tight_layout()
            fig3 = plt.figure(figsize=[18, 5])
            ####### Row 0: u(t,x) ##################
            plt.subplot(131)
            plt.colorbar(plt.imshow(w_exact, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('w_exact')
            plt.subplot(132)
            plt.colorbar(plt.imshow(w_pred, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('w_pred')
            plt.subplot(133)
            plt.colorbar(plt.imshow(np.abs(w_pred - w_exact), interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('w_error')
            plt.tight_layout()
            fig4 = plt.figure(figsize=[18, 5])
            ####### Row 0: u(t,x) ##################
            plt.subplot(131)
            plt.colorbar(plt.imshow(p_exact, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('p_exact')
            plt.subplot(132)
            plt.colorbar(plt.imshow(p_pred, interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('p_pred')
            plt.subplot(133)
            plt.colorbar(plt.imshow(np.abs(p_pred - p_exact), interpolation='nearest', cmap='jet',
                                    extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin='lower', aspect='auto'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('p_error')
            plt.tight_layout()

    MSEsum_hist = model.losssum
    MSEsum_hisp = model.lossres
    MSEsum_hisk = model.lossbc
    MSEsum_hism = model.lossin



    # number=len(lost)
    plt.figure(figsize=(8, 4))
    plt.plot(MSEsum_hist, linewidth='2', color='blue')
    plt.plot(MSEsum_hisp, linewidth='2', color='red')
    plt.plot(MSEsum_hisk, linewidth='2', color='green')
    plt.plot(MSEsum_hism, linewidth='2', color='orange')

    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.title("loss")
    plt.yscale('log')
    plt.legend(['Sum', 'Residual', 'Boundary','init'])

    figx, ax = plt.subplots()  # 子图
    data = [MSEsum_hist, MSEsum_hisp, MSEsum_hisk,MSEsum_hism]
    ax.boxplot(data)
    ax.set_xticklabels(["loss_sum", "loss_res", "loss_bc", "loss_in",])  # 设置x轴刻度标签
    plt.grid(linestyle="--", alpha=0.8)
    plt.ylabel("loss")
    plt.yscale('log')
    plt.show()





