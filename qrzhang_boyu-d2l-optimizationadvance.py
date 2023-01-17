import os

if os.getcwd().split('/')[-1] != "OptimizationCode":

    os.chdir("/kaggle/input/boyu-d2l-optimization-advance-dataset/OptimizationAdvanceKaggle/Code/OptimizationCode")



%matplotlib inline

import sys

sys.path.append("..")

import d2lzh_pytorch as d2l

import torch



eta = 0.4



def f_2d(x1, x2):

    return 0.1 * x1 ** 2 + 2 * x2 ** 2



def gd_2d(x1, x2, s1, s2):

    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)



d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
eta = 0.6

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
def momentum_2d(x1, x2, v1, v2):

    v1 = gamma * v1 + eta * 0.2 * x1

    v2 = gamma * v2 + eta * 4 * x2

    return x1 - v1, x2 - v2, v1, v2



eta, gamma = 0.4, 0.5

d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
eta = 0.6

d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
features, labels = d2l.get_data_ch7()



def init_momentum_states():

    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)

    v_b = torch.zeros(1, dtype=torch.float32)

    return (v_w, v_b)



def sgd_momentum(params, states, hyperparams):

    for p, v in zip(params, states):

        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data

        p.data -= v.data
d2l.train_ch7(sgd_momentum, init_momentum_states(),

              {'lr': 0.02, 'momentum': 0.5}, features, labels)
d2l.train_ch7(sgd_momentum, init_momentum_states(),

              {'lr': 0.02, 'momentum': 0.9}, features, labels)
d2l.train_ch7(sgd_momentum, init_momentum_states(),

              {'lr': 0.004, 'momentum': 0.9}, features, labels)
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},

                    features, labels)
import os

if os.getcwd().split('/')[-1] != "OptimizationCode":

    os.chdir("/kaggle/input/boyu-d2l-optimization-advance-dataset/OptimizationAdvanceKaggle/Code/OptimizationCode")



%matplotlib inline

import math

import torch

import sys

sys.path.append("..") 

import d2lzh_pytorch as d2l



def adagrad_2d(x1, x2, s1, s2):

    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度

    s1 += g1 ** 2

    s2 += g2 ** 2

    x1 -= eta / math.sqrt(s1 + eps) * g1

    x2 -= eta / math.sqrt(s2 + eps) * g2

    return x1, x2, s1, s2



def f_2d(x1, x2):

    return 0.1 * x1 ** 2 + 2 * x2 ** 2



eta = 0.4

d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
eta = 2

d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
features, labels = d2l.get_data_ch7()



def init_adagrad_states():

    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)

    s_b = torch.zeros(1, dtype=torch.float32)

    return (s_w, s_b)



def adagrad(params, states, hyperparams):

    eps = 1e-6

    for p, s in zip(params, states):

        s.data += (p.grad.data**2)

        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)
d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
import os

if os.getcwd().split('/')[-1] != "OptimizationCode":

    os.chdir("/kaggle/input/boyu-d2l-optimization-advance-dataset/OptimizationAdvanceKaggle/Code/OptimizationCode")



%matplotlib inline

import math

import torch

import sys

sys.path.append("..") 

import d2lzh_pytorch as d2l



def rmsprop_2d(x1, x2, s1, s2):

    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6

    s1 = gamma * s1 + (1 - gamma) * g1 ** 2

    s2 = gamma * s2 + (1 - gamma) * g2 ** 2

    x1 -= eta / math.sqrt(s1 + eps) * g1

    x2 -= eta / math.sqrt(s2 + eps) * g2

    return x1, x2, s1, s2



def f_2d(x1, x2):

    return 0.1 * x1 ** 2 + 2 * x2 ** 2



eta, gamma = 0.4, 0.9

d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
features, labels = d2l.get_data_ch7()



def init_rmsprop_states():

    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)

    s_b = torch.zeros(1, dtype=torch.float32)

    return (s_w, s_b)



def rmsprop(params, states, hyperparams):

    gamma, eps = hyperparams['gamma'], 1e-6

    for p, s in zip(params, states):

        s.data = gamma * s.data + (1 - gamma) * (p.grad.data)**2

        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)
d2l.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9},

              features, labels)
d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
def init_adadelta_states():

    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    return ((s_w, delta_w), (s_b, delta_b))



def adadelta(params, states, hyperparams):

    rho, eps = hyperparams['rho'], 1e-5

    for p, (s, delta) in zip(params, states):

        s[:] = rho * s + (1 - rho) * (p.grad.data**2)

        g =  p.grad.data * torch.sqrt((delta + eps) / (s + eps))

        p.data -= g

        delta[:] = rho * delta + (1 - rho) * g * g
d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)
d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
import os

if os.getcwd().split('/')[-1] != "OptimizationCode":

    os.chdir("/kaggle/input/boyu-d2l-optimization-advance-dataset/OptimizationAdvanceKaggle/Code/OptimizationCode")



%matplotlib inline

import torch

import sys

sys.path.append("..") 

import d2lzh_pytorch as d2l



features, labels = d2l.get_data_ch7()



def init_adam_states():

    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    return ((v_w, s_w), (v_b, s_b))



def adam(params, states, hyperparams):

    beta1, beta2, eps = 0.9, 0.999, 1e-6

    for p, (v, s) in zip(params, states):

        v[:] = beta1 * v + (1 - beta1) * p.grad.data

        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2

        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])

        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])

        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)

    hyperparams['t'] += 1
d2l.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)