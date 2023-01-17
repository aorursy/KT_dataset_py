! pip uninstall -y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c ai-project-life-environment
!unzip ai-project-life-environment.zip
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
train = pd.read_csv('train data.csv')

train
xtrain = train.loc[:, [i for i in train.keys()[1:-1]]]
ytrain = train[train.keys()[-1]]

xtrain = np.array(xtrain)
ytrain = np.array(ytrain).reshape(-1,1)


xtrain = torch.FloatTensor(xtrain).to(device)
ytrain = torch.FloatTensor(ytrain).to(device)
xtrain
#random seed
torch.manual_seed(1)
random.seed(1)

#hidden layer
lin1 = nn.Linear(7,4)
lin2 = nn.Linear(4,1)

nn.init.xavier_uniform_(lin1.weight)
nn.init.xavier_uniform_(lin2.weight)

relu = nn.ReLU()


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
test = pd.read_csv('test data.csv')

xtest = test.loc[:, [i for i in test.keys()[1:]]]
xtest = np.array(xtest)

xtest = torch.from_numpy(xtest).float().to(device)

H = model(xtest)

predic = H.cpu().detach().numpy().reshape(-1,1)

submit = pd.read_csv('submit sample.csv')
for i in range(len(submit)):
  submit['Expected'][i] = predic[i]

submit
submit.to_csv('submit.csv',mode='w',index=False)
! kaggle competitions submit -c ai-project-life-environment -f submit.csv -m "submit"