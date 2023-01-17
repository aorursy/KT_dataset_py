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
from sklearn.preprocessing import MinMaxScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
lr=0.1
epochs=50
batch_size=500
train_data=pd.read_csv('/kaggle/input/stay-or-leave/train_data.csv',header=None,skiprows=1,usecols=range(0,10))
test_data=pd.read_csv('/kaggle/input/stay-or-leave/test_data.csv',header=None,skiprows=1,usecols=range(0,9))
x_train=train_data.loc[:,0:8]
y_train=train_data.loc[:,[9]]

x_train=np.array(x_train)
y_train=np.array(y_train)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
data_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)
linear1=torch.nn.Linear(9,5,bias=True)
linear2=torch.nn.Linear(5,5,bias=True)
linear3=torch.nn.Linear(5,1,bias=True)
sig=torch.nn.Sigmoid()
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
model=torch.nn.Sequential(linear1,sig,
                          linear2,sig,
                          linear3).to(device)
loss=torch.nn.BCELoss().to(device) 
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
total_batch=len(data_loader)
for epoch in range(epochs):
  avg_cost=0

  for X,Y in data_loader:
    X=X.to(device)
    Y=Y.float().to(device)

    optimizer.zero_grad()
    hypothesis=model(X)
    cost=loss(sig(hypothesis), Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/total_batch

  print('epoch:','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost))
print('learning finished')
with torch.no_grad():
  x_test=test_data.loc[:,:]
  x_test=np.array(x_test)
  x_test=scaler.transform(x_test)
  x_test=torch.FloatTensor(x_test).to(device)

  predict=model(x_test)
predict[predict>0.5350]=1
predict[predict<=0.5350]=0
prediction=predict.long().cpu().numpy().reshape(-1,1)
id=np.array([i for i in range(len(predict))]).reshape(-1,1).astype(np.uint32)

result = np.hstack([id,prediction])

submit= pd.DataFrame(result, columns=('Id','Expected'))