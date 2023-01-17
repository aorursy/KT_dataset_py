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
train = pd.read_csv('/kaggle/input/2020ai-project-18011797/waterquality_train.csv')
test = pd.read_csv('/kaggle/input/2020ai-project-18011797/waterquality_test.csv')
submission = pd.read_csv('/kaggle/input/2020ai-project-18011797/waterquality_submit.csv')
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn import preprocessing

import torchvision.datasets as data
import torchvision.transforms as transforms
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device =='cuda':
  torch.cuda.manual_seed_all(777)

#batch size를 67로 해서 drop_last할 때 버려지는 데이터 최소화
learning_rate = 1e-3
batch_size = 67
drop_prob = 0.3

#전처리 StandardScaler 사용
scaler = preprocessing.StandardScaler()
train=train.values[:,:]

x_train=train[:,1:-1]
y_train=train[:,[-1]]

x_train = scaler.fit_transform(x_train)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

train_dataset = torch.utils.data.TensorDataset(x_train,y_train)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#layer 2개
linear1 = torch.nn.Linear(7,7,bias=True)
linear2 = torch.nn.Linear(7,1,bias=True)

#활성화 함수 ReLU 사용
relu = torch.nn.ReLU()

#xavier_uniform initialization 
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)

dropout = torch.nn.Dropout(p=drop_prob)
model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#회귀 모델이므로 MSELoss 함수 적용
loss = torch.nn.MSELoss().to(device)
total_batch = len(data_loader)

for epoch in range(20000):
    avg_cost = 0
    model.train()
    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = loss(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    if epoch%1000==0:
      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(avg_cost))

with torch.no_grad():
  model.eval()
  x_test = test.values[:,1:]
  x_test = np.array(x_test)
  x_test = scaler.transform(x_test)
  x_test = torch.from_numpy(x_test).float().to(device)
  
  prediction = model(x_test)
prediction = prediction.cpu().numpy().reshape(-1,1)
for i in range(len(prediction)):
    submission['Expected'][i] = prediction[i].item()

# int형 변환을 하여 csv에 저장
submission['Expected']=submission['Expected'].astype(int)

# submission 결과값은 아래와 같음
submission