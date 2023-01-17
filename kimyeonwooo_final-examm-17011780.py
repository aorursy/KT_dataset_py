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
learning_rate = 0.1

training_epoch = 1000

batch_size = 50
train[0] = train[0] % 10000 /100

train.drop(4, axis=1,inplace=True) #rainfall 삭제



xtrain = train.loc[:,[i for i in train.keys()[:-1]]]

ytrain = train[train.keys()[-1]]



xtrain = np.array(xtrain)

xtrain = torch.FloatTensor(xtrain).to(device)



ytrain = np.array(ytrain)

ytrain = torch.FloatTensor(ytrain).view(-1,1).to(device)



train
train_dataset = torch.utils.data.TensorDataset(xtrain,ytrain)



data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size = batch_size,

                                          shuffle=True,

                                          drop_last=True)
xtrain.shape
lin1 = nn.Linear(6,12)

lin2 = nn.Linear(12,24)

lin3 = nn.Linear(24,12)

lin4 = nn.Linear(12,6)

lin5 = nn.Linear(6,1)



nn.init.kaiming_uniform_(lin1.weight)

nn.init.kaiming_uniform_(lin2.weight)

nn.init.kaiming_uniform_(lin3.weight)

nn.init.kaiming_uniform_(lin4.weight)

nn.init.kaiming_uniform_(lin5.weight)



relu = nn.ReLU()



model = nn.Sequential(lin1,relu,

                      lin2,relu,

                      lin3,relu,

                      lin4,relu,

                      lin5

                      ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss = nn.MSELoss().to(device)
total_batch = len(data_loader)



for epoch in range(training_epoch):

    avg_cost = 0

    for X,Y in data_loader:

        X = X.to(device)

        Y = Y.to(device)



        optimizer.zero_grad()

        hypothesis = model(X)

        cost = loss(hypothesis,Y)

        cost.backward()

        optimizer.step()





        avg_cost += cost/total_batch

    

    if epoch % 10 == 0:  

        print('Epoch:', '%d' % (epoch ), 'Cost =', '{:.9f}'.format(avg_cost))

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



submit.to_csv('submission.csv', index = None, header=True)