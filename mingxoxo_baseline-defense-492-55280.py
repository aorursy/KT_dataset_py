import torch

import pandas as pd

import numpy as np

from sklearn import preprocessing



import random

from torch.utils.data import DataLoader, TensorDataset



torch.manual_seed(777)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(777)
scaler = preprocessing.StandardScaler()
train = pd.read_csv('../input/2020-ai-termproject-18011793/train.csv', header=None, skiprows=1)

test = pd.read_csv('../input/2020-ai-termproject-18011793/test.csv', header=None, skiprows=1)
train[0] = train[0]%10000/100

x_train = train.loc[:,0:9]

y_train = train.loc[:,[10]]



x_train = np.array(x_train)

y_train = np.array(y_train)

x_train = scaler.fit_transform(x_train)



x_train = torch.FloatTensor(x_train).to(device)

y_train = torch.FloatTensor(y_train).to(device)
y_train
dataset = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)



#model = torch.nn.Linear(10,1).to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.8)

#loss = torch.nn.MSELoss().to(device)
linear1 = torch.nn.Linear(10,512,bias=True)

linear2 = torch.nn.Linear(512,512,bias=True)

linear3 = torch.nn.Linear(512,1,bias=True)

relu = torch.nn.LeakyReLU()
torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

torch.nn.init.xavier_uniform_(linear3.weight)
model = torch.nn.Sequential(linear1,relu,

                        linear2,relu,

                        linear3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

loss = torch.nn.MSELoss().to(device)
epochs = 1000

cost_list = []

for epoch in range(epochs+1):

  for x,y in dataloader:



    x = x.to(device)

    y = y.to(device)



    hypothesis = model(x)

    cost = loss(hypothesis, y)



    optimizer.zero_grad()

    cost.backward()

    optimizer.step()



  if epoch%100 == 0:

    print('Epoch {} Cost {}'.format(epoch, cost.item()))
with torch.no_grad():

  test[0] = test[0]%10000/100

  x_test = test.loc[:,:]

  x_test = np.array(x_test)

  x_test = scaler.transform(x_test)

  x_test = torch.from_numpy(x_test).float().to(device)



  p = model(x_test)
p = p.cpu().numpy().reshape(-1, 1)

submit = pd.read_csv('submit_sample.csv')

for i in range(len(p)):

  submit['Total'][i]=p[i].item()

submit
submit.to_csv('submit.csv', index=False, header=True)

!kaggle competitions submit -c 2020-ai-termproject-18011793 -f submit.csv -m "defense"