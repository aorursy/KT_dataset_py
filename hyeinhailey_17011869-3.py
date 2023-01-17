import torch

import torch.nn as nn

import random

import pandas as pd

import numpy as np

from sklearn import preprocessing



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(111)



train = pd.read_csv('../input/2020-ai-exam-fashionmnist-3/mnist_train_label.csv', header=None)

train
from torch.utils.data import DataLoader, TensorDataset



xtrain = train.loc[:, 1:784]

ytrain = train.loc[:, 0]



print(xtrain.shape, ytrain.shape)



xtrain=np.array(xtrain)

ytrain=np.array(ytrain)



Scaler = preprocessing.MinMaxScaler()

xtrain = Scaler.fit_transform(xtrain)



xtrain=torch.FloatTensor(xtrain).to(device)

ytrain=torch.LongTensor(ytrain).to(device)



dataset = TensorDataset(xtrain, ytrain)

dl = DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

random.seed(111)

torch.manual_seed(111)



epochs=15



lin1 = nn.Linear(784, 512, bias=True)

lin2 = nn.Linear(512, 512,bias=True)

lin3 = nn.Linear(512, 10, bias=True)

#nn.init.kaiming_uniform(lin1.weight)

#nn.init.kaiming_uniform(lin2.weight)

#nn.init.kaiming_uniform(lin3.weight)

relu = nn.ReLU()



model = nn.Sequential(lin1, relu,

                      lin2, relu,

                      lin3).to(device)



loss = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



for epoch in range(epochs):

  avg_cost=0



  for x,y in dl:

    x = x.to(device)

    y = y.to(device)



    optimizer.zero_grad()

    H = model(x)

    cost = loss(H, y)

    cost.backward()

    optimizer.step()



    avg_cost += cost/len(dl)



  print('Epoch:', '%04d'%(epoch+1) ,'Cost : {:.9f}'.format(avg_cost))



print('Finished')
test = pd.read_csv('../input/2020-ai-exam-fashionmnist-3/mnist_test.csv', header=None, usecols=range(1,785))

test = test.fillna(0)

test


with torch.no_grad():

  xtest=test.loc[:,:]

  xtest=np.array(xtest)

  xtest=Scaler.transform(xtest)

  xtest=torch.FloatTensor(xtest).to(device)



  H = model(xtest)

  correct_prediction = torch.argmax(H, 1)



correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)



submit=pd.read_csv('../input/2020-ai-exam-fashionmnist-3/submission.csv')



for i in range(len(correct_prediction)):

  submit['Category'][i]=correct_prediction[i].item()



submit.to_csv('sub.csv',index=False, header=True)

submit