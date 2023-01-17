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

import torchvision.datasets as data

import torchvision.transforms as transforms

import random

from torch.utils.data import  TensorDataset, DataLoader

import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정

torch.manual_seed(777)  # seed 777

random.seed(777)

torch.cuda.manual_seed_all(777)



learning_rate = 0.00003

training_epochs = 200

batch_size =20

drop_prob = 0.3

scaler = MinMaxScaler()  # 표준화를 standard에서 minmax로 바꿈
train_data=pd.read_csv('train_AI_project.csv').dropna()  #nan은 드랍



test_data=pd.read_csv('test_AI_porject.csv').dropna()

train_data['Year']=train_data['Year']%10000/100  # 연도 데이터를 월.일로 바꿈

x_train_data=train_data.loc[:,[i for i in train_data.keys()[:-1]]]

y_train_data=train_data[train_data.keys()[7]]



x_train_data=np.array(x_train_data)

y_train_data=np.array(y_train_data)

x_train_data = scaler.fit_transform(x_train_data)



x_train_data=torch.FloatTensor(x_train_data)

y_train_data=torch.FloatTensor(y_train_data)



# nn을 사용하기위해 데이터셋을 맞춰줌

train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)



data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          drop_last=True)
linear1 = torch.nn.Linear(7,256,bias=True)  # 3중 nn

linear2 = torch.nn.Linear(256,256,bias=True)

linear3 = torch.nn.Linear(256,1,bias=True)



relu = torch.nn.LeakyReLU() #leakyReLU로 바꿈

dropout = torch.nn.Dropout(p=drop_prob)  #dropout을 해줌



torch.nn.init.kaiming_normal_(linear1.weight)

torch.nn.init.kaiming_normal_(linear2.weight)

torch.nn.init.kaiming_normal_(linear3.weight)
model = torch.nn.Sequential(linear1,relu,dropout,

                            linear2,relu,dropout,

                            linear3).to(device)
# 손실함수와 최적화 함수

loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam사용
total_batch = len(data_loader)

model.train()

for epoch in range(training_epochs):

    avg_cost = 0



    for X, Y in data_loader:



        X = X.to(device) #gpu사용

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
with torch.no_grad(): #gpu사용안할떄



  #위의 트레인 방식과 같음

  test_data['Year']=test_data['Year']%10000/100

  x_test_data=test_data.loc[:,[i for i in test_data.keys()[:]]]

  x_test_data=np.array(x_test_data)

  x_test_data = scaler.transform(x_test_data)

  x_test_data=torch.from_numpy(x_test_data).float().to(device)



  prediction = model(x_test_data)

    

correct_prediction = prediction.cpu().numpy().reshape(-1,1)



submit=pd.read_csv('submit_sample_AI_project.csv')



# Expected란에 예측한 데이터 넣어줌

for i in range(len(correct_prediction)):  

  submit['Expected'][i]=correct_prediction[i].item()



submit.to_csv('submit.csv', mode = 'w', index = False, header = True)   #submit.csv파일로 예측했던값 넣어줬던 데이터를 만듬

!kaggle competitions submit -c ai-project-foodpoisoning -f submit.csv -m "16013074 김영진" #제출