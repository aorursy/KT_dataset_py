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
from sklearn import preprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 1e-3
training_epochs = 20000
batch_size = 140
drop_prob = 0.3
Scaler = preprocessing.StandardScaler()
xy = pd.read_csv('/kaggle/input/library-project/library_train.csv',header=None)
xy = xy.loc[2:181,3:12]
# object형식 -> float형식
xy = xy.astype(float)
# 모든 변인
x_data = xy.loc[:,3:11]
x_data = np.array(x_data)
y_data = xy.loc[:,12]

# 정규화
x_data = Scaler.fit_transform(x_data)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(np.array(y_data))

xy_test = pd.read_csv('/kaggle/input/library-project/library_test.csv',header=None)
xy_test = xy_test.loc[2:73,3:11]
xy_test = xy_test.astype(float)
x_test = xy_test
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(9,9,bias=True)
linear2 = torch.nn.Linear(9,1,bias=True)


relu = torch.nn.LeakyReLU()
dropout = torch.nn.Dropout(p=drop_prob)

torch.nn.init.kaiming_uniform_(linear1.weight)
torch.nn.init.kaiming_uniform_(linear2.weight)



model = torch.nn.Sequential(linear1,relu, dropout,
                            linear2).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)

for epoch in range(20000):
    avg_cost = 0

    model.train()
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
    if epoch%1000==0:
      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
with torch.no_grad():
  model.eval()
  x_test = np.array(x_test)
  x_test = Scaler.transform(x_test)
  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('/kaggle/input/library-project/sample_submit.csv')

cnt=0
for i in range(len(correct_prediction)):
  submit['Expected'][i] = correct_prediction[i]
  submit['id'][i] = cnt
  cnt +=1
submit