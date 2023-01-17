import torch

import pandas as pd

import numpy as np

from sklearn import preprocessing

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(777)



train = pd.read_csv('../input/ai-tomato/training_set.csv', header=None, skiprows=1)



train
train[0] = train[0] % 10000 /100

xtrain = train.loc[:,[i for i in train.keys()[:-1]]]

ytrain = train[train.keys()[-1]]



xtrain = np.array(xtrain)

xtrain = torch.FloatTensor(xtrain).to(device)



ytrain = np.array(ytrain)

ytrain = torch.FloatTensor(ytrain).view(-1,1).to(device)



ytrain
dataset = TensorDataset(xtrain, ytrain)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



torch.manual_seed(1)



model = nn.Linear(7,1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8)

loss = nn.MSELoss().to(device)



nb_epochs = 500

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
test = pd.read_csv('../input/ai-tomato/test_set.csv')

test=test.dropna(axis=1)
test['date'] = test['date'] % 10000 /100

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