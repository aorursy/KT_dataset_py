%load_ext autoreload

%autoreload 2

%matplotlib inline

import torch
from pathlib import Path

import requests

import pickle

import gzip

from torch import tensor
DATA_PATH = Path("data")

PATH = DATA_PATH / "mnist"



PATH.mkdir(parents=True, exist_ok=True)



URL = "http://deeplearning.net/data/mnist/"

FILENAME = "mnist.pkl.gz"



if not (PATH / FILENAME).exists():

        content = requests.get(URL + FILENAME).content

        (PATH / FILENAME).open("wb").write(content)
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:

        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

        x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
def normalize(x, m, s): return (x-m)/s
train_mean,train_std = x_train.mean(),x_train.std()

train_mean,train_std
n,c = x_train.shape

x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()
import pickle, gzip, math, torch, matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rcParams['image.cmap'] = 'gray'
img = x_train[0]

img.view(28,28).type()
plt.imshow(img.view((28,28)))
n,m = x_train.shape

c = y_train.max()+1

nh = 50

n,m,c, nh
w1 = torch.randn(m,nh)/math.sqrt(m)

b1 = torch.zeros(nh)

w2 = torch.randn(nh,1)/math.sqrt(nh)

b2 = torch.zeros(1)
def lin(x, w, b): return x@w + b

def relu(x): return x.clamp_min(0.)
# Doing transformation here on x_valid.

t = relu(lin(x_valid, w1, b1))
def model(xb):

    l1 = lin(xb, w1, b1)

    l2 = relu(l1)

    l3 = lin(l2, w2, b2)

    return l3
%timeit -n 10 _=model(x_valid)
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
# Calculating the mse for our initial model as built above,

y_train,y_valid = y_train.float(),y_valid.float()

mse(model(x_train),y_train)
def mse_grad(inp, targ): 

    # grad of loss with respect to output of previous layer

    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
# Defining gradient for relu layer:

def relu_grad(inp, out):

    # grad of relu with respect to input activations

    # this is 1 when input is greater than zero else 0.

    inp.g = (inp>0).float() * out.g
# Defining linear gradient

def lin_grad(inp, out, w, b):

    inp.g = out.g @ w.t()

    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)

    b.g = out.g.sum(0)
# Implementing chain rule of derivatives will help in performing the back propagation, here is the full implementation of code.
def forward_and_backward(inp, targ):

    # forward pass:

    l1 = inp @ w1 + b1

    l2 = relu(l1)

    out = l2 @ w2 + b2

    # loss calculation after forward pass.

    loss = mse(out, targ)

    

    # backward pass:

    mse_grad(out, targ)

    lin_grad(l2, out, w2, b2)

    relu_grad(l1, l2)

    lin_grad(inp, l1, w1, b1)
# Lets run forward and backward once and see what happens.

forward_and_backward(x_train, y_train)
# We are saving to check our implementation and check with Pytorch standard implementation later on.

w1g = w1.g.clone()

w2g = w2.g.clone()

b1g = b1.g.clone()

b2g = b2.g.clone()

ig  = x_train.g.clone()
xt2 = x_train.clone().requires_grad_(True)

w12 = w1.clone().requires_grad_(True)

w22 = w2.clone().requires_grad_(True)

b12 = b1.clone().requires_grad_(True)

b22 = b2.clone().requires_grad_(True)
def forward(inp, targ):

    # forward pass:

    l1 = inp @ w12 + b12

    l2 = relu(l1)

    out = l2 @ w22 + b22

    # we don't actually need the loss in backward!

    return mse(out, targ)
loss = forward(xt2, y_train)
loss.backward()
# Lets look at some of the gradients now.

w22.grad
w2g