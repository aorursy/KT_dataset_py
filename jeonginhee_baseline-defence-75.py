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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  torch.cuda.manual_seed_all(777)

random.seed(777)
torch.manual_seed(777)

Scaler=preprocessing.StandardScaler()

xy_data=pd.read_csv('train.csv')
print(xy_data.info())
print(xy_data.isna().sum())
xy_data=xy_data.fillna(0)
xy_data.head()
xy_data.mean()['solar']
test_data=pd.read_csv('test.csv')
test_data
import datetime as dt

xy_data=pd.read_csv('train.csv')
xy_data['solar'] = xy_data['solar'].interpolate()
xy_data.fillna(value={'rain':0}, inplace=True)

xy_data['date'] = pd.to_datetime(xy_data['date'])
xy_data['date'] = xy_data['date'].map(dt.datetime.toordinal)
xy_data
test_data = pd.read_csv('test.csv')
test_data['solar'] = test_data['solar'].interpolate()
test_data.fillna(value={'rain':0}, inplace=True)

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['date'] = test_data['date'].map(dt.datetime.toordinal)
test_data
xy_data.isna().sum()
test_data.isna().sum()
x_data=xy_data.loc[:,'aveTemp':'solar']
y_data=xy_data.loc[:,'Category']
x_data=np.array(x_data,dtype=float)
y_data=np.array(y_data,dtype=float)

x_data=Scaler.fit_transform(x_data)
x_train=torch.FloatTensor(x_data).to(device)
y_train=torch.LongTensor(y_data).to(device)
train_dataset=torch.utils.data.TensorDataset(x_train, y_train)
# 학습 파라미터 설정
learning_rate = 0.1
batch_size = 100
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1=torch.nn.Linear(5,6,bias=True)
linear2=torch.nn.Linear(6,6,bias=True)
linear3=torch.nn.Linear(6,6,bias=True)
linear4=torch.nn.Linear(6,6,bias=True)
linear5=torch.nn.Linear(6,4,bias=True)
relu=torch.nn.ELU()
torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
torch.nn.init.kaiming_normal_(linear5.weight)
model=torch.nn.Sequential(linear1,relu,
                          linear2,relu,
                          linear3,relu,
                          linear4,relu,
                          linear5
                          ).to(device)
loss = torch.nn.CrossEntropyLoss().to(device) # softmax 내부적으로 계산
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(350+1):
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

    if epoch%100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
with torch.no_grad():
    test=test_data.loc[:,'aveTemp':'solar']
    test=np.array(test,dtype=float)
    test=Scaler.transform(test)
    test=torch.from_numpy(test).float().to(device)
    prediction = model(test)
    correct_prediction = torch.argmax(prediction,dim=1)

