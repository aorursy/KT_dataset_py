import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

data.head(3)
from mxnet import nd

import mxnet

from mxnet.gluon import nn as nm

import mxnet.gluon as gluon

import pandas as pd

import seaborn as sns

import numpy as np

import os

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

#import torchsummary

import torch

from torch import nn

#import torchvision

#from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import History 

from keras.utils import plot_model

from keras.optimizers import SGD

from keras.utils import np_utils

from keras.layers.core import Dense, Dropout, Activation



data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

data_test.head()
labels= {0 :'T-shirt/top', 1 :'Trouser', 2:'Pullover', 3:'Dress' ,4:'Coat', 5:'Sandal' ,6:'Shirt', 7:'Sneaker',8:'Bag' ,9 :'Ankle_boot'}
X = data_train.drop('label',axis =1)

y= data_train['label']

batch_size = 64

t =mxnet.nd.array(X)

l=mxnet.nd.array(y)
X1 = data_test.drop('label',axis =1)

y1= data_test['label']

batch_size = 64

t1 =mxnet.nd.array(X1)

l2=mxnet.nd.array(y1)
y=y.values

X=X.values

y1=y1.values

X1=X1.values
Y_train = np_utils.to_categorical(y, 10)

Y_train1 = np_utils.to_categorical(y1, 10)
N, D_in, H, D_out = 64, 784, 150, 10

model1 =Sequential()

model1.add(Dense(units=H, activation='relu', input_dim=D_in))

model1.add(Dense(units=H, activation='tanh'))

model1.add(Dense(units=H, activation='tanh'))

model1.add(Dense(units=D_out, activation='softmax'))



model1.summary()
optimizer=keras.optimizers.SGD(lr=.1)

model1.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

model1.fit(X, Y_train,batch_size=64, epochs=200,verbose=2,validation_data=(X1, Y_train1))