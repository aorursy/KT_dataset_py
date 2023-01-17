# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import torch

from torch import nn,functional

from torch.autograd import Variable

import torchvision

import torchvision.transforms as transforms



from sklearn.model_selection import train_test_split



import os
#Loading the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

X = train.loc[:,train.columns != "label"].values/255   #Normalizing the values

Y = train.label.values



features_train, features_test, targets_train, targets_test = train_test_split(X,Y,test_size=0.2,

                                                                              random_state=42)

X_train = torch.from_numpy(features_train)

X_test = torch.from_numpy(features_test)



Y_train = torch.from_numpy(targets_train).type(torch.LongTensor) 

Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)



train = torch.utils.data.TensorDataset(X_train,Y_train)

test = torch.utils.data.TensorDataset(X_test,Y_test)





train_loader = torch.utils.data.DataLoader(train, batch_size = train_batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = test_batch_size, shuffle = True)
dataiter = iter(train_loader)

x, y = dataiter.next()



print(x.shape)

print(y.shape)