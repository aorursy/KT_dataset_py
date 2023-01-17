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
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화
from sklearn import preprocessing
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from torch.utils.data import  TensorDataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 1000
batch_size = 200
drop_prob = 0.3
xy_train = pd.read_csv('train.csv', header = None, skiprows=1)
xy_train[0] = xy_train[0] % 10000 / 100
x_data = xy_train.loc[:, 0:4]
y_data = xy_train.loc[:, [5]]
x_data = np.array(x_data)
y_data = np.array(y_data)

scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device) 
train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)
linear1 = torch.nn.Linear(5, 8,bias=True)
linear2 = torch.nn.Linear(8, 16,bias=True)
linear3 = torch.nn.Linear(16, 32,bias=True)
linear4 = torch.nn.Linear(32, 32,bias=True)
linear5 = torch.nn.Linear(32, 32,bias=True)
linear6 = torch.nn.Linear(32, 32,bias=True)
linear7 = torch.nn.Linear(32, 32,bias=True)
linear8 = torch.nn.Linear(32, 16,bias=True)
linear9 = torch.nn.Linear(16, 8,bias=True)
linear10 = torch.nn.Linear(8, 1,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p = 0.3)

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

model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5, relu, dropout,
                            linear6, relu, dropout,
                            linear7, relu, dropout,
                            linear8, relu, dropout,
                            linear9, relu, dropout,
                            linear10).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []
model_history = []
err_history = []

total_batch = len(data_loader)
model.train()

for epoch in range(training_epochs + 1):
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

  model_history.append(model)
  err_history.append(avg_cost)
  
  if epoch % 10 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test.csv', header = None, skiprows=1)
xy_test[0] = xy_test[0] % 10000 / 100
x_data = xy_test.loc[:, :]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()  # 주의사항 (dropout=False)
    
    predict = best_model(x_test)
submit = pd.read_csv('sample.csv')
submit['Expected'] = submit['Expected'].astype(float)
for i in range(len(predict)):
  submit['Expected'][i] = predict[i]
submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
