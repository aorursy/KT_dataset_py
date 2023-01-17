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

training_epochs = 1000                      #------------- epochs 변경 40,000 -> 1000

batch_size = 80                             #------------- batch_size 변경 

# drop_prob = 0.0                           

Scaler = preprocessing.MinMaxScaler()       #------------- MinMaxScaler 사용
# 데이터 로드

xy = pd.read_csv('library_train.csv',header=None)

xy = xy.loc[2:181,3:12]

# object형식 -> float형식

xy = xy.astype(float)

# 모든 변인

x_data = xy.loc[:,3:11]

x_data = np.array(x_data)

y_data = xy.loc[:,[12]]     #------------------- y 열벡터로 변경 [12]

# 정규화

x_data = Scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data)

y_train = torch.FloatTensor(np.array(y_data))

# 테스트 데이터 로드

xy_test = pd.read_csv('library_test.csv',header=None)

xy_test = xy_test.loc[2:73,3:11]

xy_test = xy_test.astype(float)

x_test = xy_test

x_test = np.array(x_test)

x_test = Scaler.transform(x_test)

x_test = torch.from_numpy(x_test).float().to(device)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          drop_last=True)
# layer 2개

linear1 = torch.nn.Linear(9,9,bias=True)

linear2 = torch.nn.Linear(9,9,bias=True)

linear3 = torch.nn.Linear(9,4,bias=True)

linear4 = torch.nn.Linear(4,4,bias=True)

linear5 = torch.nn.Linear(4,1,bias=True)  #----------- NN재설계





relu = torch.nn.ReLU()                    #----------- ReLU()로 변경

dropout = torch.nn.Dropout(p=drop_prob)   #----------- dropout = 0



## Random Init => Xavier Init

torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(linear4.weight)

torch.nn.init.xavier_uniform_(linear5.weight)





model = torch.nn.Sequential(linear1,relu, dropout,

                            linear2,relu, dropout,

                            linear3,relu, dropout,

                            linear4,relu, dropout,

                            linear5

                            ).to(device)
loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
# 모델 학습

total_batch = len(data_loader)

model.train()

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



    if epoch % 100 == 0:

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost))



print('Learning finished')