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
import torch

import torch.nn as nn

import pandas as pd

import numpy as np

import random

import math

from sklearn.preprocessing import MinMaxScaler



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(111)
train = pd.read_csv('../input/fired-area-prediction/forestfires_train.csv')



train
train.drop('rain', axis=1, inplace=True) #delete rain column



train
xtrain = train.loc[:, [i for i in train.keys()[4:-1]]]

ytrain = train[train.keys()[-1]]



xtrain = np.array(xtrain)

ytrain = np.array(ytrain).reshape(-1,1)



scaler = MinMaxScaler()

xtrain = scaler.fit_transform(xtrain) #normalization



xtrain = torch.FloatTensor(xtrain).to(device)

ytrain = torch.FloatTensor(ytrain).to(device)
xtrain #scaler result
ytrain.min()
ytrain.max()
for i in range(len(ytrain)): #area <- ln(x+1) transform

  ytrain[i] = math.log(ytrain[i]+1)



ytrain #ln(x+1) transform result
#random seed

torch.manual_seed(1)

random.seed(1)



#hidden layer

lin1 = nn.Linear(7,4)

lin2 = nn.Linear(4,1)



nn.init.xavier_uniform_(lin1.weight)

nn.init.xavier_uniform_(lin2.weight)



relu = nn.ReLU()

dropout = nn.Dropout(p = 0.3) #prevent overfitting



#model

model = nn.Sequential(lin1, relu, dropout,

                      lin2).to(device)



epochs = 15000

lr = 1e-4



loss = nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)



for epoch in range(epochs+1):

  H = model(xtrain)

  cost = loss(H, ytrain)



  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch % 1000 == 0:

    print('Epoch:', '%05d'%epoch, 'Cost: {:.5f}'.format(cost.item()))



print('Finished')
test = pd.read_csv('../input/fired-area-prediction/forestfires_test.csv')

test.drop('rain', axis=1, inplace=True) #delete rain column



xtest = test.loc[:, [i for i in test.keys()[4:]]]

xtest = np.array(xtest)



xtest = scaler.transform(xtest)

xtest = torch.from_numpy(xtest).float().to(device)



H = model(xtest)



#inverse ln(x+1) transform

for i in range(len(H)):

  H[i] = torch.exp(H[i]) - 1



predic = H.cpu().detach().numpy().reshape(-1,1)



submit = pd.read_csv('../input/fired-area-prediction/forestfires_submission.csv')

for i in range(len(submit)):

  submit['prediction'][i] = predic[i]



submit