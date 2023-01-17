# 기존 코드와 차별점 : 
# 파라미터 변경  
# NN 재설계  
# 데이터 처리방법 변경  
# 레이어 추가
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
# 학습 파라미터 설정
learning_rate = 0.001
training_epochs = 50
batch_size = 100
train=pd.read_csv('/kaggle/input/2020ai-project-18011797/waterquality_train.csv',header=None, skiprows=1)

train[0] = train[0]%10000/100

data_x = train.loc[:, 0:7]
data_y = train.loc[:,[8]]

data_x = np.array(data_x)
data_y = np.array(data_y)

# 스케일러 활용
Scaler = preprocessing.StandardScaler()
data_x = Scaler.fit_transform(data_x)

x_train = torch.FloatTensor(data_x)
y_train = torch.FloatTensor(data_y)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
# 레이어 생성
linear1 = torch.nn.Linear(8,512, bias = True)
linear2 = torch.nn.Linear(512,512, bias = True)
linear3 = torch.nn.Linear(512,512, bias = True)
linear4 = torch.nn.Linear(512,512, bias = True)
linear5 = torch.nn.Linear(512,1, bias = True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)
model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5).to(device)
# 손실함수와 최적화 함수
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

    print('Epoch:', '%03d' % (epoch + 1), 'rate =', '{:.1f}'.format(avg_cost))

print('Learning finished')
test_xy = pd.read_csv("/kaggle/input/2020ai-project-18011797/waterquality_test.csv", header=None, skiprows=1)
test_xy[0] = test_xy[0]%10000/100

with torch.no_grad():

  x_test_data=test_xy.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('/kaggle/input/2020ai-project-18011797/waterquality_submit.csv')
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()
submit.to_csv('submit.csv',index=False,header=True)