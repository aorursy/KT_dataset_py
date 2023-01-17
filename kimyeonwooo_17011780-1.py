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
import numpy as np

import pandas as pd

import torch

import torch.optim as optim

import numpy as np

import pandas as pd

import torch.nn.functional as F

import random

from sklearn import preprocessing

import torch.nn as nn

import torchvision.datasets as data

import torchvision.transforms as transforms

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

import torch

import torchvision.datasets as dsets

import torchvision.transforms as transforms

import torch.nn.init

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import time

import torch.nn.functional as F

import torch.nn as nn

import matplotlib.pyplot as plt

 

from torchvision import models



device = 'cuda' if torch.cuda.is_available() else 'cpu'



if device == 'cuda':

  torch.cuda.manual_seed_all(777)



random.seed(777)

torch.manual_seed(777)
train=np.loadtxt("../input/2020-ai-exam-fashionmnist-1/mnist_train_label.csv",delimiter=',')

train
train.shape
x_train=torch.from_numpy(train[:,1:])

y_train=torch.from_numpy(train[:,0])
test=np.loadtxt("../input/2020-ai-exam-fashionmnist-1/mnist_test.csv",delimiter=',')

test
len(test)
x_train = torch.FloatTensor(np.array(x_train)).to(device)

y_train = torch.LongTensor(np.array(y_train)).to(device) 
test = torch.FloatTensor(np.array(test)).to(device)
x_train.shape
y_train.shape
y_train=y_train.view(-1)
test.shape
learning_rate = 0.001

training_epochs = 15

batch_size = 200
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)



data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          drop_last=False)
linear1 = torch.nn.Linear(784,10,bias=True)



relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
model = torch.nn.Sequential(linear1

                            ).to(device)
loss = torch.nn.CrossEntropyLoss().to(device) 

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
total_batch = len(data_loader)

model_h = []

error_h = []

for epoch in range(training_epochs):

  avg_cost = 0



  for X, Y in data_loader:



        X = X.to(device)

        Y = Y.to(device)

  

     

        optimizer.zero_grad()

        hypothesis = model(X)

        

        cost = loss(hypothesis, Y)

        cost.backward()

        optimizer.step()

        avg_cost += cost

        avg_cost /= total_batch

        model_h.append(model)

        error_h.append(avg_cost)



  if epoch % 1000 == 0 :

        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))



print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
best_model = model_h[np.argmin(error_h)]
with torch.no_grad():

  model.eval()



  prediction = best_model(test)

  prediction=torch.argmax(prediction,dim=1) #다중분류



prediction = prediction.cpu().numpy().reshape(-1,1)
len(prediction)
submit=pd.read_csv('../input/2020-ai-exam-fashionmnist-1/submission.csv')

submit
id=np.array([i for i in range(len(prediction))]).reshape(-1,1)

result=np.hstack([id,prediction])

submit=pd.DataFrame(result,columns=["ID","Category"])

submit
submit.to_csv('submission.csv',index=False,header=True)