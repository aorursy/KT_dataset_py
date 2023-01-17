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
import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)

if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 1e-4
training_epoches = 100
batch_size = 50
Scaler = preprocessing.StandardScaler()
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.drop(['snowFall','deepSnowfall'], axis=1, inplace=True)
test_data.drop(['snowFall','deepSnowfall'], axis=1, inplace=True)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

x_train = train_data.loc[:,'avgTemp':'fogDuration']
y_train = train_data['trafficAccident']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

print(x_train)
print(y_train)


train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last=True)
linear1 = torch.nn.Linear(6,64, bias = True) # feature
linear2 = torch.nn.Linear(64,64, bias = True)
linear3 = torch.nn.Linear(64,32, bias = True)
linear4 = torch.nn.Linear(32,32, bias = True)
linear5 = torch.nn.Linear(32,32, bias = True)
linear6 = torch.nn.Linear(32,32, bias = True)
linear7 = torch.nn.Linear(32,16, bias = True)
linear8 = torch.nn.Linear(16,16, bias = True)
linear9 = torch.nn.Linear(16,8, bias = True)
linear10 = torch.nn.Linear(8,8, bias = True)
linear11 = torch.nn.Linear(8,4, bias = True)
linear12 = torch.nn.Linear(4,1, bias = True)


torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)
torch.nn.init.xavier_uniform_(linear8.weight)
torch.nn.init.xavier_uniform_(linear9.weight)
torch.nn.init.xavier_uniform_(linear10.weight)
torch.nn.init.xavier_uniform_(linear11.weight)
torch.nn.init.xavier_uniform_(linear12.weight)

model = torch.nn.Sequential(linear1,
                            linear2,
                            linear3,
                            linear4,
                            linear5,
                            linear6,
                            linear7,
                            linear8,
                            linear9,
                            linear10,
                            linear11,
                            linear12
                            ).to(device)

loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

for epoch in range(training_epoches):
  avg_cost = 0

  for X, Y in data_loader:

    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
  
  print('Epoch:','%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
print('Learning finshed')

with torch.no_grad():
  x_test = test_data.loc[:,'avgTemp':'fogDuration']
  x_test = np.array(x_test)
  #x_test = Scaler.transform(x_test)
  x_test = torch.from_numpy(x_test).float()

  prediction = model(x_test)
  correct_prediction = prediction.cpu().numpy().reshape(-1,1)
 

submit = pd.read_csv('submit_sample.csv')

for i in range(len(correct_prediction)):
  submit['Expected'][i] = correct_prediction[i].item()

submit
submit.to_csv('submit.csv', mode='w', index = False)
!kaggle competitions submit -c traffic-accident -f submit.csv -m "Message"