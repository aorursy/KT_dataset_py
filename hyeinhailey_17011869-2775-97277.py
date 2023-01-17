import torch

import pandas as pd

import numpy as np

from sklearn import preprocessing

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

import random



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(777)



train = pd.read_csv('../input/predict-number-of-asthma-patient/train_disease.csv', header=None, skiprows=1)



train


xtrain = train.loc[:,[i for i in train.keys()[1:-1]]]

ytrain = train[train.keys()[-1]]



scaler = preprocessing.MinMaxScaler()

xtrain = np.array(xtrain)

xtrain = scaler.fit_transform(xtrain) #스케일 조정

xtrain = torch.FloatTensor(xtrain).to(device)



ytrain = np.array(ytrain)

ytrain = torch.FloatTensor(ytrain).view(-1,1).to(device)



xtrain
#random seed 설정

torch.manual_seed(1)

random.seed(1)



dataset = TensorDataset(xtrain, ytrain)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)



model = nn.Linear(4,1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) #옵티마이저 SGD

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
test = pd.read_csv('../input/predict-number-of-asthma-patient/test_disease.csv')

test['Data'] = test['Data'] % 10000 / 100

xtest = test.loc[:,[i for i in test.keys()[1:]]]

xtest = np.array(xtest)

xtest = scaler.transform(xtest)

xtest = torch.from_numpy(xtest).float().to(device)



xtest
H = model(xtest)



H = H.cpu().detach().numpy().reshape(-1,1)



submit = pd.read_csv('../input/predict-number-of-asthma-patient/submission.csv')



for i in range(len(submit)):

  submit['Expect'][i] = H[i].astype(int)



submit.to_csv('sub.csv', index = None, header=True)



submit