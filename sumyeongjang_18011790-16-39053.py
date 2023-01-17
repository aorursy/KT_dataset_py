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
import torch.nn as nn
import math
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 0.1
training_epochs = 100
batch_size = 1
Scaler = MinMaxScaler()

train_data=pd.read_csv('/kaggle/input/fired-area-prediction/forestfires_train.csv',header=None,skiprows=1,usecols=range(4,13))
test_data=pd.read_csv('/kaggle/input/fired-area-prediction/forestfires_test.csv',header=None,skiprows=1,usecols=range(4,12))

x_train_data=train_data.loc[:,0:11]
y_train_data=train_data.loc[:,12]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)

print(x_train_data)

train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
linear1 = torch.nn.Linear(8,8,bias=True)
linear2 = torch.nn.Linear(8,1,bias=True)

relu = torch.nn.ReLU()
# Random Init => Xavier Init
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
for epoch in range(training_epochs):
  avg_cost = 0

  for X, Y in data_loader:
    
    X = X.to(device)
    Y = Y.to(device)

    # 그래디언트 초기화
    optimizer.zero_grad()
    # Forward 계산
    hypothesis = model(X)
    # Error 계산
    cost = loss(hypothesis, Y)
    # Backparopagation
    cost.backward()
    # 가중치 갱신
    optimizer.step()

    # 평균 Error 계산
    avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
  print('Learning finished')
# Test the model using test sets
with torch.no_grad():

  x_test_data=test_data.loc[:,0:11]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)

correct_prediction = prediction.cpu().numpy().reshape(-1,1)

submit=pd.read_csv('/kaggle/input/fired-area-prediction/forestfires_submission.csv')

for i in range(len(correct_prediction)):
  submit['prediction'][i]=correct_prediction[i].item()
submit

submit.to_csv('submission.csv',index=False,header=True)
!kaggle competitions submit -c fired-area-prediction -f submission.csv -m "Message"