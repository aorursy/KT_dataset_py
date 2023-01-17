import torch

import pandas as pd

import numpy as np

from sklearn import preprocessing

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(111)



train = pd.read_csv('../input/ai-tomato/training_set.csv', header=None, skiprows=1)



train
train[0] = train[0] % 10000 /100

train.drop(4, axis=1,inplace=True) #rainfall 삭제



xtrain = train.loc[:,[i for i in train.keys()[:-1]]]

ytrain = train[train.keys()[-1]]



xtrain = np.array(xtrain)

xtrain = torch.FloatTensor(xtrain).to(device)



ytrain = np.array(ytrain)

ytrain = torch.FloatTensor(ytrain).view(-1,1).to(device)



train
dataset = TensorDataset(xtrain, ytrain)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)



torch.manual_seed(111)



lin1 = nn.Linear(6,32)

lin2 = nn.Linear(32,1)



nn.init.kaiming_uniform_(lin1.weight)

nn.init.kaiming_uniform_(lin2.weight)



relu = nn.ReLU()



model = nn.Sequential(lin1,relu,

                      lin2).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

loss = nn.MSELoss().to(device)



nb_epochs = 500

for epoch in range(nb_epochs + 1):

  for x,y in dataloader:

    x = x.to(device)

    y=y.to(device)



    H = model(x)

    cost = loss(H, y)



    optimizer.zero_grad()

    cost.backward()

    optimizer.step()



  if epoch%50 == 0:

      print('Epoch {}  Cost {}'.format(epoch, cost.item()))



print('Learning Finished')
test = pd.read_csv('../input/ai-tomato/test_set.csv')

test
test=test.dropna(axis=1)
test['date'] = test['date'] % 10000 /100

test.drop('rain fall', axis=1,inplace=True) #rainfall 삭제



xtest = test.loc[:,[i for i in test.keys()[:]]]

xtest = np.array(xtest)

xtest = torch.from_numpy(xtest).float().to(device)



H = model(xtest)



H = H.cpu().detach().numpy().reshape(-1,1)



submit = pd.read_csv('../input/ai-tomato/submit_sample.csv')



for i in range(len(submit)):

  submit['expected'][i] = H[i]



submit.to_csv('sub.csv', index = None, header=True)



submit