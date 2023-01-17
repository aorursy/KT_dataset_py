!pip install torchtext==0.2.3
!pip install fastai==0.7.0
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt, rcParams, animation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from fastai.imports import *
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *
from fastai.torch_imports import *
from fastai.io import *
import torch.nn as nn
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#data label
labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def display_all(df):
    with pd.option_context("display.max_rows", 100, "display.max_columns", 100): 
        display(df)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(labels[int(title)])

def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(labels[int(titles[i])], fontsize=16)
        plt.imshow(ims[i], cmap='gray')
%load_ext autoreload
%autoreload 2
%matplotlib inline
PATH = '../input/'
!ls {PATH}

df_raw = pd.read_csv(f'{PATH}fashion-mnist_train.csv', low_memory=False)
display_all(df_raw.head().T)
# Randomly sampling the dataset
df_raw = df_raw.sample(frac=1)
# Setting the validation size to 10,000
n_valid = 10000
n_train = len(df_raw) - n_valid
y, x = df_raw['label'].values, df_raw.loc[:, df_raw.columns != 'label'].values
x_train, x_valid = split_vals(x, n_train)
y_train, y_valid = split_vals(y, n_train)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
mean = x_train.mean()
std = x_train.std()

x_train=(x_train-mean)/std
mean, std, x_train.mean(), x_train.std()
# To maintain consistency we subtract and divide the validation set data with the mean and standard deviation of training data
x_valid = (x_valid-mean)/std
x_valid.mean(), x_valid.std()
x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape
np.unique(y_train), np.unique(y_valid)
show(x_imgs[14], y_valid[14])
x_imgs[0,10:15,10:15]
show(x_imgs[0,10:15,10:15])
plots(x_imgs[:10], titles=y_valid[:10])
# loss function
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
from fastai.dataset import *
md = ImageClassifierData.from_arrays(PATH, (x_train,y_train), (x_valid, y_valid))
import torch.nn as nn
net = nn.Sequential(
    nn.Linear(28*28, 10),
    nn.LogSoftmax()
).cuda()
loss=nn.NLLLoss()
metrics=[accuracy]
# setting learning rate as 0.1
opt=optim.SGD(net.parameters(), 1e-1)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)
preds = predict(net, md.val_dl)
preds.shape
preds.argmax(axis=1)[:5]
preds = preds.argmax(1)
np.mean(preds == y_valid)
plots(x_imgs[:8], titles=preds[:8])
def get_weights(*dims): return nn.Parameter(torch.randn(dims)/dims[0])
def softmax(x): return torch.exp(x)/(torch.exp(x).sum(dim=1)[:,None])

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = (x @ self.l1_w) + self.l1_b  # Linear Layer
        x = torch.log(softmax(x)) # Non-linear (LogSoftmax) Layer
        return x
net2 = LogReg().cuda()
opt=optim.SGD(net2.parameters(), 1e-2)
fit(net2, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
preds = predict(net2, md.val_dl).argmax(1)
plots(x_imgs[35:45], titles=preds[35:45])
np.mean(preds == y_valid)
opt=optim.SGD(net2.parameters(), 1e-2, weight_decay=1e-3)
fit(net2, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
preds = predict(net2, md.val_dl).argmax(1)
plots(x_imgs[35:45], titles=preds[35:45])
np.mean(preds == y_valid)
net3 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-2
optimizer=optim.SGD(net3.parameters(), lr=learning_rate)
def score(x, y):
    y_pred = to_np(net3(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)
# Fit function unboxed
def train(no_of_epochs=10):
    for epoch in range(no_of_epochs):
        losses=[]
        dl = iter(md.trn_dl)
        for t in range(len(md.trn_dl)):
            # Forward pass: compute predicted y and loss by passing x to the model.
            xt, yt = next(dl)
            y_pred = net3(V(xt))
            l = loss(y_pred, V(yt))
            losses.append(l)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            l.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        val_dl = iter(md.val_dl)
        val_scores = [score(*next(val_dl)) for i in range(len(md.val_dl))]
        print(np.mean(val_scores))
train(1)
train(3)
net4 = LogReg().cuda()
loss_fn = nn.NLLLoss()
lr = 1e-2
w,b = net4.l1_w, net4.l1_b
def score(x, y):
    y_pred = to_np(net4(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)
def train(no_epochs):
    for epoch in range(no_epochs):
        losses = []
        dl = iter(md.trn_dl)
        for t in range(len(md.trn_dl)):
            xt, yt = next(dl)
            y_pred = net4(V(xt))

            # compute the loss
            l = loss(y_pred, Variable(yt).cuda())
            losses.append(loss)
            
            # computing gradient of the loss with respect to parameter
            l.backward()

            # updating the parameters with gradient
            w.data -= w.grad.data * lr
            b.data -= b.grad.data * lr
            
            #Initialize the gradient to zeros
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        val_dl = iter(md.val_dl)
        val_scores = [score(*next(val_dl)) for i in range(len(md.val_dl))]
        print(np.mean(val_scores))
train(1)
train(3)
