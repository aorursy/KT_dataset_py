%load_ext autoreload

%autoreload 2

%matplotlib inline

import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import operator



#export

from pathlib import Path

from IPython.core.debugger import set_trace

from fastai import datasets

import pickle, gzip, math, torch, matplotlib as mpl

import matplotlib.pyplot as plt

from torch import tensor

import pandas as pd



MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'



def get_data():

    path = datasets.download_data(MNIST_URL, ext='.gz')

    with gzip.open(path, 'rb') as f:

        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    return map(tensor, (x_train, y_train, x_valid, y_valid))



def normalize(x,m,s): return (x-m)/s
x_train, y_train, x_valid, y_valid = get_data()

print(x_train.mean(), x_train.std())

train_mean, train_valid = x_train.mean(), x_train.std()



x_train = normalize(x_train, train_mean, train_valid)

x_valid = normalize(x_valid, train_mean, train_valid) # Use train, not valid

print(x_train.mean(), x_train.std())
def test_near_zero(a, tol=1e-3): assert a.abs()<tol, f"Near zero:{a}"



test_near_zero(x_train.mean())

test_near_zero(1- x_train.std())
n,m = x_train.shape

c   = y_train.max()+1

nh  = 50

n,m,c,nh
# kaiming init / he init

w1 = torch.randn(m,nh)*math.sqrt(1./m)

b1 = torch.zeros(nh)

w2 = torch.randn(nh,1)*math.sqrt(1./nh)

b2 = torch.zeros(1)



test_near_zero(w1.mean())

test_near_zero(w1.std()-1/math.sqrt(m))
x_valid.mean(), x_valid.std() # Normalized
def lin(x,w,b): return x@w+b

def relu(x): return x.clamp_min(0.)



t = lin(x_valid,w1,b1); print(t.mean(), t.std())

t = relu(t)           ; print(t.mean(), t.std())

# relu's mean would be a half, not zero, which is explained later
# kaiming init / he init for relu

w1 = torch.randn(m,nh)*math.sqrt(2./m); print(w1.mean(),w1.std())

t  = relu(lin(x_valid,w1,b1))        ; print( t.mean(), t.std())
# replacing with pytorch init

from torch.nn import init

w1 = torch.zeros(m,nh)

init.kaiming_normal_(w1, mode='fan_out'); print(w1.mean(), w1.std())

t  = relu(lin(x_valid,w1,b1))           ; print( t.mean(),   t.std())

# In this case, {'fan_in':'fan_out' = m:nh}
from torch import nn

w1.shape, nn.Linear(m,nh).weight.shape # transposed
def relu(x): return x.clamp_min(0.)-0.5

w1 = torch.randn(m,nh)*math.sqrt(2./m)

t1 = relu(lin(x_valid,w1,b1)); t1.mean(), t1.std()

# relu's mean is now near zero
def model(xb): return lin(relu(lin(xb,w1,b1)), w2, b2)

model(x_valid).shape
def mse(outp, targ): return (outp.squeeze()-targ).pow(2).mean()

y_train, y_valid = y_train.float(), y_valid.float()

mse(model(x_train), y_train)
def mse_grad(inp, targ): # inp is previous output == pred

    inp.g =  2.*(inp.squeeze()-targ).unsqueeze(-1) / y_train.shape[0]



def relu_grad(inp, out):

    inp.g = (inp>0).float() * out.g



def lin_grad(inp, out, w, b):

    inp.g = out.g @ w.t()

    w.g   = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)

    b.g   = out.g.sum(0)



def fwd_bk(inp, targ):

    l1 = inp @ w1 + b1

    l2 = relu(l1)

    out = l2 @ w2 + b2

    loss = mse(out, targ)

    

    mse_grad(out, targ)

    lin_grad(l2, out, w2, b2)

    relu_grad(l1, l2)

    lin_grad(inp, l1, w1, b1)
fwd_bk(x_train, y_train)



w1g = w1.g.clone(); w2g = w2.g.clone()

b1g = b1.g.clone(); b2g = b2.g.clone()

ig  = x_train.g.clone()



# PyTorch can manage above backprop process

xt2 = x_train.clone().requires_grad_(True)

w12 = w1.clone().requires_grad_(True)

w22 = w2.clone().requires_grad_(True)

b12 = b1.clone().requires_grad_(True)

b22 = b2.clone().requires_grad_(True)
def forward(inp, targ): return mse(relu(inp @ w12 + b12) @ w22 + b22, targ)



loss = forward(xt2, y_train)

loss.backward()
def test(a,b,cmp,cname=None):

    if cname is None: cname=cmp.__name__

    assert cmp(a,b), f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq, '==')

def near(a,b): return torch.allclose(a,b,1e-3,1e-5)

def test_near(a,b): test(a,b,near)



test_near(w22.grad, w2g)

test_near(b22.grad, b2g)

test_near(w12.grad, w1g)

test_near(b12.grad, b1g)

test_near(xt2.grad, ig)
# Refactoring

# I'm skipping
class Model(nn.Module):

    def __init__(self, n_in, nh, n_out):

        super().__init__()

        self.layers = [nn.Linear(n_in, nh), nn.ReLU(), nn.Linear(nh, n_out)]

        self.loss = mse

    def __call__(self, x, targ):

        for l in self.layers: x = l(x)

        return self.loss(x.squeeze(), targ)
%time loss = Model(m, nh, 1)(x_train, y_train)
%time loss.backward()