import pandas as pd

import numpy as np

import torch

import random

from sklearn import preprocessing 

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(777)





train = pd.read_csv('../input/traffic-accident/train.csv', usecols=range(3,12))

test = pd.read_csv('../input/traffic-accident/test.csv', usecols=range(3,11)) 



#비어있는 값이 많은 열 삭제

train.drop(['snowFall','deepSnowfall','fogDuration'], axis=1, inplace=True)

test.drop(['snowFall','deepSnowfall','fogDuration'], axis=1, inplace=True)



train = train.fillna(0)



test = test.fillna(0)

test
xtrain = train.loc[:,[i for i in train.keys()[:-1]]]

ytrain = train[train.keys()[-1]]



xtrain = np.array(xtrain)

#xtrain = Scaler.fit_transform(xtrain) <-정규화 하지 않음

xtrain = torch.FloatTensor(xtrain).to(device)



ytrain = np.array(ytrain)

ytrain = torch.FloatTensor(ytrain).view(-1,1).to(device)



print(xtrain.shape, ytrain.shape)



xtrain
dataset = TensorDataset(xtrain, ytrain)



dataloader = DataLoader(dataset, batch_size=64, shuffle=True)





random.seed(1)

torch.manual_seed(1)

torch.cuda.manual_seed_all(1)



lin1 = nn.Linear(5,5, bias = True)

lin2 = nn.Linear(5,5, bias = True)

lin3 = nn.Linear(5,1, bias = True)



nn.init.xavier_uniform_(lin1.weight)

nn.init.xavier_uniform_(lin2.weight)

nn.init.xavier_uniform_(lin3.weight)



relu = nn.ReLU()

model = nn.Sequential(lin1,

                      lin2,

                      lin3).to(device)





optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss = nn.MSELoss().to(device)



nb_epochs = 300



for epoch in range(nb_epochs + 1):

  for batch_idx, samples in enumerate(dataloader):

    x, y = samples



    H = model(x)

    cost = loss(H, y)



    optimizer.zero_grad()

    cost.backward()

    optimizer.step()



  if epoch%50 == 0:

      print('Epoch {}  Cost {}'.format(epoch, cost.item()))



print('Learning Finished')
with torch.no_grad(): 

  xtest = test.loc[:,[i for i in train.keys()[:-1]]]



  xtest = np.array(xtest)



  #xtest = Scaler.transform(xtest)



  xtest = torch.from_numpy(xtest).float().to(device)



  H = model(xtest)



  correct_prediction = H.cpu().numpy().reshape(-1,1) 
submit = pd.read_csv('../input/traffic-accident/submit_sample.csv')



for i in range(len(correct_prediction)):

  submit['Expected'][i] = correct_prediction[i]
