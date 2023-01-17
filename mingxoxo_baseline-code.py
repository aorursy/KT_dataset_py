import torch

import torch.optim as optim

import numpy as np

import pandas as pd



#학습시킬 데이터를 pandas를 통해 확인

xy = pd.read_csv("../input/2020-ai-termproject-18011793/train.csv")

xy
#데이터 전처리(year과 스케일이 큰 9to10:23to24, total 값들을 소수점 자리들로 변경)

xy['year'] = xy['year']%10000/100

xy.loc[:,'9to10':'23to24'] = xy.loc[:,'9to10':'23to24']/100

xy['total'] = xy['total']/1000

xy
#Tensor형 데이터로 만들기

x_p = xy.loc[:, "year": "maxTemp"]

y_p = xy["total"]

x_data = np.array(x_p)

y_data = np.array(y_p)

x_train = torch.FloatTensor(x_data)

y_train = torch.FloatTensor(y_data).unsqueeze(1)
#W, b값 학습 (배추값과 같은 다른 회귀 문제와 학습과정 동일)



W = torch.zeros((10, 1), requires_grad= True)

b = torch.zeros(1, requires_grad= True)



optimizer = optim.SGD([W, b], lr = 1e-4)



nb_epochs = 10000



for epoch in range(nb_epochs + 1):



  hypothesis = x_train.matmul(W) + b



  cost = torch.mean((hypothesis - y_train)**2)



  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch % 1000 == 0 :

    print(epoch, cost.item())
#test데이터를 받아 위에서와 마찬가지로 데이터 전처리

test = pd.read_csv("../input/2020-ai-termproject-18011793/test.csv")

test['year'] = test['year']%10000/100

test.loc[:,'9to10':'23to24'] = test.loc[:,'9to10':'23to24']/100



x_test = test.loc[:, "year": "maxTemp"]

test_x = torch.FloatTensor(np.array(x_test))



predict = (test_x.matmul(W) + b).detach()



#학습시킬 때 xy['total']/1000 이렇게 사용하였으므로 다시 1000을 곱해주어 원래 값을 예측

predict = predict*1000 
#submission.csv 생성

submit = pd.read_csv("../input/2020-ai-termproject-18011793/submit_sample.csv")

for i in range(len(predict)):

  submit["Total"][i] = predict[i].int()

submit["Total"] =submit["Total"].astype(int)



submit.to_csv("submission.csv", index = False, header = True)