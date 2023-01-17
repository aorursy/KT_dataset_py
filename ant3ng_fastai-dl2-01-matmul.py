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

path = datasets.download_data(MNIST_URL, ext='.gz');path
def test(a,b,cmp,cname=None):

    if cname is None: cname=cmp.__name__

    assert cmp(a,b), f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq, '==')
with gzip.open(path, 'rb') as f:

    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')



x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))

n,c = x_train.shape

plt.imshow(x_train[0].view(28, -1))
w = torch.randn(784, 10)

b = torch.zeros(10)
def mm(a,b):

    ar,ac = a.shape

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar,bc)

    for i in range(ar):

        for j in range(bc):

            for k in range(ac):

                c[i,j] += a[i,k] * b[k,j]

    return c



m1 = x_valid[:5]

%time t1 = mm(m1, w); t1.shape
def mm(a,b):

    ar,ac = a.shape

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar,bc)

    for i in range(ar):

        for j in range(bc):

            c += (a[i,:] * b[:,j]).sum()

    return c



%time t1 = mm(m1, w)
def near(a,b): return torch.allclose(a,b,1e-3,1e-5)

def test_near(a,b): test(a,b,near)



test_near(t1, mm(m1,w))
def mm(a,b):

    ar,ac = a.shape

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar,bc)

    for i in range(ar):

        c[i] = (a[i].unsqueeze(-1) * b).sum()

    return c



%time t1 = mm(m1, w)
def mm(a,b): return torch.einsum('ik,kj -> ij', a,b)

%time t1=mm(m1,w)
%time t1=m1@w