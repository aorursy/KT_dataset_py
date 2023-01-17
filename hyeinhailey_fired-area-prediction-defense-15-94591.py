import torch

import torch.nn.functional as F

import pandas as pd

import numpy as np

from sklearn import preprocessing,utils

import math

import random



device='cuda' if torch.cuda.is_available() else 'cpu'



if device=='cuda':

  torch.cuda.manual_seed_all(111)





train=pd.read_csv('../input/fired-area-prediction/forestfires_train.csv')

train=utils.shuffle(train)

train
#Since 'FFMC' values are left skewed

for i in range(len(train)):

  train['FFMC'][i] = (train['FFMC'][i])**3



train['FFMC']
#strings -> numerical value

encoder = preprocessing.LabelEncoder()

encoder.fit(train['month'])

train['month'] = encoder.transform(train['month'])



xtrain=train.loc[:, [i for i in train.keys()[0:-1]]]

ytrain=train[train.keys()[-1]]



#add interactive data

xtrain['FFMC.DMC'] = xtrain['FFMC']*xtrain['DMC']

xtrain['FFMC.DC'] = xtrain['FFMC']*xtrain['DC']

xtrain['DMC.DC'] = xtrain['DMC']*xtrain['DC']

xtrain['temp.RH'] = xtrain['RH']*xtrain['temp']



xtrain.drop(['X','Y','day', 'FFMC','temp','RH', 'wind', 'rain'], axis=1, inplace=True)



xtrain
xtrain=np.array(xtrain)

ytrain=np.array(ytrain).reshape(-1,1)



print(xtrain.shape)

print(ytrain.shape)



scaler = preprocessing.MinMaxScaler()

xtrain = scaler.fit_transform(xtrain)



xtrain=torch.FloatTensor(xtrain).to(device)

ytrain=torch.FloatTensor(ytrain).to(device)



for i in range(len(ytrain)):

  ytrain[i] = math.log(ytrain[i]+1)



xtrain
torch.manual_seed(111)

random.seed(111)



lin1 = torch.nn.Linear(8,6)

lin2 = torch.nn.Linear(6,3)

lin3 = torch.nn.Linear(3,1)



torch.nn.init.kaiming_uniform_(lin1.weight)

torch.nn.init.kaiming_uniform_(lin2.weight)

torch.nn.init.kaiming_uniform_(lin3.weight)



relu = torch.nn.ReLU()

dropout = torch.nn.Dropout(p=0.25)

model = torch.nn.Sequential(lin1,relu,dropout,

                            lin2,relu,dropout,

                            lin3).to(device)



epochs = 15000

lr = 1e-4



optimizer=torch.optim.Adam(model.parameters(), lr=lr)



for epoch in range(epochs+1):

  H = model(xtrain)

  cost = F.mse_loss(H, ytrain).to(device)



  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch % 1000 == 0:

    print('Epoch: %05d'%epoch,' Cost {:.5f}'.format(cost.item()))
torch.nn.init.xavier_uniform_(lin3.weight)

test=pd.read_csv('../input/fired-area-prediction/forestfires_test.csv')



#unseen label(labelencoder)

for i in range(len(test)):

  for label in np.unique(test['month'][i]):

    if label not in encoder.classes_:

      encoder.classes_ = np.append(encoder.classes_, label)

    

test['month'] = encoder.transform(test['month'])



for i in range(len(test)):

  test['FFMC'][i] = test['FFMC'][i]**3



xtest = test.loc[:, [i for i in test.keys()[0:]]]



#add interactive data

xtest['FFMC.DMC'] = xtest['FFMC']*xtest['DMC']

xtest['FFMC.DC'] = xtest['FFMC']*xtest['DC']

xtest['DMC.DC'] = xtest['DMC']*xtest['DC']

xtest['temp.RH'] = xtest['RH']*xtest['temp']



xtest.drop(['X','Y','day','FFMC','temp','RH', 'wind', 'rain'], axis=1, inplace=True)



xtest
xtest=np.array(xtest)



xtest = scaler.transform(xtest) 



xtest=torch.FloatTensor(xtest).to(device)



H=model(xtest)



for i in range(len(H)):

  H[i]=torch.exp(H[i])-1



result = pd.read_csv('../input/fired-area-prediction/forestfires_submission.csv')

for i in range(len(result)):

  result['prediction'][i] = H[i].item()



result.to_csv('defense.csv', header=True, index=False)



result