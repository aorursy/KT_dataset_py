import pandas as pd

import torch 

import torch.nn as nn

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.autograd import Variable





# data

test  = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
print(train.shape)

train.sample(10)
X_train = torch.Tensor(train.drop(['label'], axis=1).values.astype('float32').reshape(42000,1,28,28))

print(X_train)
y_train = torch.IntTensor(train['label'].values.astype('int32'))

print(y_train)