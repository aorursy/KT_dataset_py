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
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c ai-tomato
!unzip ai-tomato.zip
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
  torch.cuda.manual_seed_all(111)

train = pd.read_csv('training_set.csv', header=None, skiprows=1)

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
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True) # 배치사이즈 5로 바꿔줌

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
test = pd.read_csv('test_set.csv')
test=test.dropna(axis=1)
test['date'] = test['date'] % 10000 /100
test.drop('rain fall', axis=1,inplace=True) #rainfall 삭제

xtest = test.loc[:,[i for i in test.keys()[:]]]
xtest = np.array(xtest)
xtest = torch.from_numpy(xtest).float().to(device)

H = model(xtest)

H = H.cpu().detach().numpy().reshape(-1,1)

submit = pd.read_csv('submit_sample.csv')

for i in range(len(submit)):
  submit['expected'][i] = H[i]

submit.to_csv('sub.csv', index = None, header=True)

submit
!kaggle competitions submit -c ai-tomato -f sub.csv -m "Message"






