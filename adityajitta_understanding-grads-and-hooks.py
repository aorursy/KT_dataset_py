!pip install torchviz
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from fastai.vision import *

from fastai.metrics import error_rate

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import numpy as np 



import matplotlib.pyplot as plt

import time

import os

from torchviz import make_dot, make_dot_from_trace
from copy import deepcopy
import fastai
class SimpleHooks(fastai.callbacks.HookCallback):

    "Callback that record the mean and std of activations."



    def on_train_begin(self, **kwargs):

        "Initialize stats."

        super().on_train_begin(**kwargs)

        self.stats = []



    def hook(self, m:nn.Module, i:Tensors, o:Tensors):

        "Take the mean and std of `o`."

        return i,o

    def on_batch_end(self, train, **kwargs):

        "Take the stored results and puts it in `self.stats`"

        if train: self.stats.append(self.hooks.stored)

    def on_train_end(self, **kwargs):

        "Polish the final result."

        print("Hi")
from fastai.callbacks import ActivationStats

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
path = untar_data(URLs.MNIST_SAMPLE)

data = ImageDataBunch.from_folder(path)

#learn = cnn_learner(data, models.resnet18, callback_fns=ActivationStats)

learn = Learner(data, simple_cnn((3,16,16,2)), callback_fns=SimpleHooks)
m11 = learn.model[0][0]

something = []

def gradients(something):

    def some_func(x):

        something.append(x)

    return some_func

sfunc_ = gradients(something)

#hk1 = m11.weight.register_hook(sfunc_)
learn.wd = 0.0001

learn.fit(1)
[ptuple[0] for ptuple in learn.model.named_parameters()]
[mtuple[0] for mtuple in learn.model.named_modules()]
data_iter = iter(data.single_dl)

data_item = next(data_iter)
x = data_item[0]

make_dot(learn.model(x), params=dict(learn.model.named_parameters()))
x.shape
#exp_learn.model[1][0].weight.requires_grad = False

#exp_learn.model[1][0].bias.requires_grad = False

#make_dot(exp_learn.model(x), params=dict(exp_learn.model.named_parameters()))
#exp_learn.model[1][0].weight.requires_grad = True

#exp_learn.model[1][0].bias.requires_grad = True
#torch.set_grad_enabled(True)

#exp_learn.model.eval()

#for param in exp_learn.model.parameters():

#        param.requires_grad = True

#make_dot(exp_learn.model(x), params=dict(exp_learn.model.named_parameters()))
module_2 = learn.model[1][0]
learn.model
#outputs= []

#def hook1(module, input, output):

#    outputs.append(output)

#hk = learn.model[2][0].register_forward_hook(hook1)

#hk.remove()
class SimpleHookClass():

    def __init__(self, module):

        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.features = output

    def close(self):

        self.hook.remove()
#%matplotlib inline

#def show(img):

#    npimg = img.numpy()

#    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
#show(data_item[0].squeeze().cpu())
#sff = SimpleHookClass(module_2)

#h = module_2.weight.register_hook(lambda grad: grad)

#learn.model(data_item[0])
#[n for n in module_2.buffers()]
#learn.backward()
#outputs[0].detach().squeeze().cpu().shape

#plt.imshow(outputs[0].detach().squeeze()[0].cpu().numpy())

#plt.imshow(outputs[0].detach().squeeze()[1].cpu().numpy())
#sff.features.detach().squeeze().cpu().shape
#plt.imshow(sff.features[0][0].detach().squeeze().cpu().numpy())
#plt.imshow(sff.features[0][1].detach().squeeze().cpu().numpy())
#def do_plot(ax, module_):

#    im = ax.imshow(module_, cmap=plt.cm.gray, interpolation='nearest')





# prepare image and figure

#fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)) = plt.subplots(1, 8,figsize=(20,20))

#axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

#for i, ax_ in enumerate(axs):

#    Z = sff.features[0][i].detach().squeeze().cpu().numpy()

#    do_plot(ax_, Z)



#plt.show()
#sff.close()
#sff.features[0][0]
#learn.model(data_item[0])
#sff.features[0][0]
#m1 = learn.model[0][1]