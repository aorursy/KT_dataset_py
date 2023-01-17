!pip3 install ipywidgets

!jupyter nbextension enable --py --sys-prefix widgetsnbextension
#!pip install --upgrade pip

#!pip install fastai==0.7.0    ## Installed from personal Github repo to avoid numpy rounding error : 

                               ## https://forums.fast.ai/t/unfamiliar-error-when-running-learn-fit/35075/19

!pip install torchtext==0.2.3

#!pip intall numpy==1.15.1   ## attirbute error thrown due to numpy updates. Changed fastai source code though

!pip install Pillow==4.1.1

!pip install blosc
%load_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.imports import *

from fastai.torch_imports import *

from fastai.io import *
import os

import pandas as pd

import pickle

import gzip
((x, y), (x_valid, y_valid), _) = pickle.load(gzip.open('../input/mnist.pkl.gz', 'rb'), encoding='latin-1')
type(x), x.shape , type(y), y.shape
mean = x.mean()

std = x.std()



x=(x-mean)/std

mean, std, x.mean(), x.std()
x_valid = (x_valid-mean)/std

x_valid.mean(), x_valid.std()
def show(img, title=None):

    plt.imshow(img, cmap="gray")

    if title is not None: plt.title(title)
def plots(ims, figsize=(12,6), rows=2, titles=None):

    f = plt.figure(figsize=figsize)

    cols = len(ims)//rows

    for i in range(len(ims)):

        sp = f.add_subplot(rows, cols, i+1)

        sp.axis('Off')

        if titles is not None: sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i], cmap='gray')
x_valid.shape
x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape
show(x_imgs[0], y_valid[0])
y_valid.shape
y_valid[0]
x_imgs[0,10:15,10:15]
show(x_imgs[0,10:15,10:15])
plots(x_imgs[:8], titles=y_valid[:8])
from fastai.metrics import *

from fastai.model import *

from fastai.dataset import *



import torch.nn as nn
net = nn.Sequential(

    nn.Linear(28*28, 100),

    nn.ReLU(),

    nn.Linear(100, 100),

    nn.ReLU(),

    nn.Linear(100, 10),

    nn.LogSoftmax()

#).cuda()  ## For GPU

)         ## For CPU
md = ImageClassifierData.from_arrays('../input/mnist.pkl.gz', (x,y), (x_valid, y_valid))
loss=nn.NLLLoss()

metrics=[accuracy]

# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)

opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)
def binary_loss(y, p):

    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
acts = np.array([1, 0, 0, 1])

preds = np.array([0.9, 0.1, 0.2, 0.8])

binary_loss(acts, preds)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)
t = [o.numel() for o in net.parameters()]

t, sum(t)
preds = predict(net, md.val_dl)
preds.shape
preds.argmax(axis=1)[:5]
preds = preds.argmax(1)
np.mean(preds == y_valid)
plots(x_imgs[:8], titles=preds[:8])