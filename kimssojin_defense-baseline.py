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
import random
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train_data=pd.read_csv('/kaggle/input/stay-or-leave/train_data.csv',header=None,skiprows=1,usecols=range(0,10))
test_data=pd.read_csv('/kaggle/input/stay-or-leave/test_data.csv',header=None,skiprows=1,usecols=range(0,9))
x_train=train_data.loc[:,0:8]
y_train=train_data.loc[:,[9]]

x_train=np.array(x_train)
y_train=np.array(y_train)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)

x_train=torch.FloatTensor(x_train)
y_train=torch.FloatTensor(y_train)
dataset=TensorDataset(x_train,y_train)
dataloader=DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
linear1=torch.nn.Linear(9,512,bias=True)
linear2=torch.nn.Linear(512,512,bias=True)
linear3=torch.nn.Linear(512,256,bias=True)
linear4=torch.nn.Linear(256,256,bias=True)
linear5=torch.nn.Linear(256,128,bias=True)
linear6=torch.nn.Linear(128,128,bias=True)
linear7=torch.nn.Linear(128,64,bias=True)
linear8=torch.nn.Linear(64,64,bias=True)
linear9=torch.nn.Linear(64,32,bias=True)
linear10=torch.nn.Linear(32,1,bias=True)
relu=torch.nn.ReLU()

torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
torch.nn.init.kaiming_normal_(linear5.weight)
torch.nn.init.kaiming_normal_(linear6.weight)
torch.nn.init.kaiming_normal_(linear7.weight)
torch.nn.init.kaiming_normal_(linear8.weight)
torch.nn.init.kaiming_normal_(linear9.weight)
torch.nn.init.kaiming_normal_(linear10.weight)
model=torch.nn.Sequential(linear1,relu,linear2,relu,linear3,relu,linear4,relu,linear5,relu,linear6,relu,linear7,relu,linear8,relu,linear9,relu,linear10).to(device)
loss=torch.nn.MSELoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
epochs=50

for epoch in range(epochs):

  for x,y in dataloader:
    x=x.to(device)
    y=y.to(device)

    hypo=model(x)
    cost=loss(hypo,y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

  print('epoch {} cost {}'.format(epoch,cost))
x_test=test_data.loc[:,:]
x_test=np.array(x_test)
x_test=scaler.transform(x_test)
x_test=torch.FloatTensor(x_test).to(device)

predict=model(x_test)
predict[predict>0.5350]=1
predict[predict<=0.5350]=0
predict=predict.long().cpu().numpy().reshape(-1,1)
id=np.array([i for i in range(len(predict))]).reshape(-1,1).astype(np.uint32)

result = np.hstack([id,predict])

submit= pd.DataFrame(result, columns=('Id','Expected'))