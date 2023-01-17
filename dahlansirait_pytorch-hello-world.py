# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# n: batch size
# d_in: input dimension
# h: hidden layer
# d_out: output dimension
n, d_in, h, d_out = 64, 1000, 100, 10

# randomly initialize input, output & weights
x = np.random.randn(n, d_in)
y = np.random.randn(n, d_out)
w1 = np.random.randn(d_in, h)
w2 = np.random.rand(h, d_out)

lr = 1e-6

for i in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # Compute loss
    loss = np.square(y_pred - y).sum()
    print("epoch: {} Loss: {}".format(i, loss)) if i % 50 == 0 else None
    
    # Backprop to compute gradiens of w1 & w2 wrt loss
    grad_y_pred = 2 * (y_pred -y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    # Update weights
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
import torch

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# n: batch size
# d_in: input dimension
# h: hidden layer
# d_out: output dimension
n, d_in, h, d_out = 64, 1000, 100, 10

# randomly initialize input, output & weights
x = torch.randn(n, d_in, device=device, dtype=dtype)
y = torch.randn(n, d_out, device=device, dtype=dtype)
w1 = torch.randn(d_in, h, device=device, dtype=dtype)
w2 = torch.rand(h, d_out, device=device, dtype=dtype)

lr = 1e-6

for i in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    # Compute loss
    loss = (y_pred - y).pow(2).sum().item()
    print("epoch: {} Loss: {}".format(i, loss)) if i % 50 == 0 else None
    
    # Backprop to compute gradients of w1 & w2 wrt loss
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    # Update weights using gradient descent
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2