import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

import os
print(os.listdir("../input"))

%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *
import torch.nn as nn

trn, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')
trn.shape, test.shape
x0, y0 = trn.drop(['label'], axis =1).values, trn['label'].values
x0.shape, y0.shape
mean= x0.mean()
std = x0.std()
x0= (x0-mean)/std
from sklearn.model_selection import train_test_split
x,x_valid, y, y_valid = train_test_split(x0, y0, test_size=0.02,random_state=42)
x.shape, x_valid.shape, y.shape
x_trn, y_trn = torch.from_numpy(x), torch.from_numpy(y)
type(x_trn)
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax()
).cuda()
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

md = ImageClassifierData.from_arrays('tmp/tmp/', (x,y), (x_valid, y_valid))
loss=nn.NLLLoss()
metrics=[accuracy]
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)
opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)

fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)
fit(net, md, n_epochs=6, crit=loss, opt=opt, metrics=metrics)
test_data=(test-mean)/std