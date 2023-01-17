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
%reload_ext autoreload
%autoreload 2

%matplotlib inline

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = '../input/dice-d4-d6-d8-d10-d12-d20/dice'
sz = 224
from glob import glob

valid_images = glob( PATH + '/valid/**/*.jpg', recursive=True)
train_images = glob( PATH + '/train/**/*.jpg', recursive=True)
plt.imshow(plt.imread(valid_images[10]))
arch = resnet50
data = ImageClassifierData.from_paths(PATH,tfms=tfms_from_model(arch, sz) )
lr = 1e-4
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(lr, 3, cycle_len=1)
learn.unfreeze()
lrs = np.array([lr/100, lr/10, lr])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
log_preds, y= learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
preds = np.argmax(probs, axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)

plot_confusion_matrix(cm, data.classes)
learn.save('resnet50_trained')
print(accuracy_np(probs, y))
#0.9904852521408183