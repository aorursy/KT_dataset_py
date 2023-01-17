# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math

import numpy as np

import matplotlib.pyplot as plt

import torch

import torchvision

from sklearn import preprocessing



%matplotlib inline

np.random.seed(1)


import scipy.io as sio 

X_test = sio.loadmat('/kaggle/input/amptest/test.mat')

print('Information for test.mat ')

print(X_test)

#print('\nThe vaulue of amp4data:')

X_test = X_test['OutputMatrix']

type(X_test)

import scipy.io as sio 

X_train = sio.loadmat('/kaggle/input/ampdata1/amp4data.mat')

print('Information for amp4data.mat ')

print(X_train)

print('\nThe vaulue of amp4data:')

X_train = X_train['OutputMatrix']

X_train=X_train[3:7]

type(X_train)
Y_train=np.zeros(10)

Y_train[5:10]=1

Y_train=Y_train[3:7]

print(Y_train)

#0 means normal pattern, 1 means distorted pattern with 5 amplitude
index=1

if Y_train[index]==0:

    print("This is a normal pattern")

else:

    print("This is a distorted pattern")

X_train_orig = X_train[index].reshape(249,249)

X_scaled = preprocessing.scale(X_train_orig)

plt.imshow(X_scaled,plt.cm.inferno)
use_gpu = torch.cuda.is_available()

use_gpu
import torch.nn as nn

import torch.nn.functional as F
X = torch.from_numpy(X_train).type(torch.FloatTensor)

y = torch.from_numpy(Y_train).type(torch.LongTensor)

if (use_gpu):

    X,y = X.cuda(),y.cuda()
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

y_test = torch.from_numpy(Y_train).type(torch.LongTensor)

if (use_gpu):

    X_test,y_test = X_test.cuda(),y_test.cuda()
import torch.nn as nn

import torch.nn.functional as F

 

class MyClassifier(nn.Module):

    def __init__(self):

        super(MyClassifier,self).__init__()

        self.fc1 = nn.Linear(62001,540)

        self.fc2 = nn.Linear(540,2)

        

    def forward(self,x):

        x = self.fc1(x)

        x = F.tanh(x)

        x = self.fc2(x)

        return x

             

    def predict(self,x):

        pred = F.softmax(self.forward(x))

        ans = []

        for t in pred:

            if t[0]>t[1]:

                ans.append(0)

            else:

                ans.append(1)

        return torch.tensor(ans)
model = MyClassifier()

criterion = nn.CrossEntropyLoss()

if(use_gpu):

    model = model.cuda()

    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

losses = []

for i in range(epochs):

    y_pred = model.forward(X)

    loss = criterion(y_pred,y)

    losses.append(loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
model.eval()

y_pred = model.predict(X_test)

result = y_pred.cpu()

result=result.detach().numpy()

print(result)

from sklearn.metrics import accuracy_score

#print(accuracy_score(model.predict(X),y))

#print(model.predict(X))

k=[0,0,1,1]

print(accuracy_score(result,k))
model.eval()

y_pred = model(X_test)

result = y_pred.cpu()

result=result.detach().numpy()

print(result)

