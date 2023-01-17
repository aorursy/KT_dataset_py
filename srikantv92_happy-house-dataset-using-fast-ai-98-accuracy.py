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
!pip install fastai==0.7.0 --no-deps
!pip install torch==0.4.1 torchvision==0.2.1
from fastai.conv_learner import *
import h5py
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)

# code for downloading resnext50 model, uncomment it when you run it for the first time
#!wget http://files.fast.ai/models/weights.tgz
#!tar -zxvf weights.tgz
#!mkdir /opt/conda/lib/python3.6/site-packages/fastai/weights/
#!cp weights/resnext_50_32x4d.pth /opt/conda/lib/python3.6/site-packages/fastai/weights/
#!rm -rf weights weights.tgz
sz = 48
arch=resnext50
def load_dataset():
    train_data = h5py.File('../input/train_happy.h5', "r")
    x_train = np.array(train_data["train_set_x"][:]) 
    y_train = np.array(train_data["train_set_y"][:]) 

    test_data = h5py.File('../input/test_happy.h5', "r")
    x_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:]) 
    
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))
    
    return x_train, y_train, x_test, y_test
X_train, Y_train, X_test, Y_test = load_dataset()
Y_train = Y_train.T.squeeze()
Y_test = Y_test.T.squeeze()
def get_data(sz):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    return ImageClassifierData.from_arrays(
        path='tmp',
        trn=(X_train, Y_train),
        val=(X_test, Y_test),
        bs=30,
        classes=Y_train,
        tfms=tfms)
data = get_data(sz)
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf = learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
lr = 0.060
learn.fit(lr, 5, cycle_len=1, cycle_mult=3)
learn.set_data(get_data(224))
learn.fit(lr, 5, cycle_len=1, cycle_mult=3)
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)