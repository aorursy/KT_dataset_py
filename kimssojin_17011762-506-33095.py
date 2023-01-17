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
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(777)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
scaler=preprocessing.StandardScaler()
train = pd.read_csv('../input/2020-ai-termproject-18011793/train.csv', header=None, skiprows=1)
test = pd.read_csv('../input/2020-ai-termproject-18011793/test.csv', header=None, skiprows=1)
train[0] = train[0] % 10000 /100
x_train = train.loc[:,0:9]
y_train = train.loc[:,[10]]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = scaler.fit_transform(x_train)

x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

model = torch.nn.Linear(10,1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
loss = torch.nn.MSELoss().to(device)
epochs = 1000
for epoch in range(epochs + 1):
    for x,y in dataloader:
      x=x.to(device)
      y=y.to(device)

      hypo = model(x)
      cost = loss(hypo, y)

      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

    if epoch%100 == 0:
      print('Epoch {}  Cost {}'.format(epoch, cost.item()))
with torch.no_grad():
  test[0] = test[0] % 10000 /100
  x_test=test.loc[:,:]
  x_test=np.array(x_test)
  x_test=scaler.transform(x_test)
  x_test=torch.from_numpy(x_test).float().to(device)

  p = model(x_test)
p = p.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('../input/2020-ai-termproject-18011793/submit_sample.csv')
for i in range(len(p)):
  submit['Total'][i]=p[i].item()
submit.to_csv('submit.csv',index=False,header=True)
