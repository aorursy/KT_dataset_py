import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# train 데이터를 받아오기
xy_train = pd.read_csv('../input/2020-ai-term-project-18011817/train_seoul_grandpark.csv', header=None, skiprows=1)
xy_train
# 데이터 프로세싱 (날짜, 미세먼지 농도, 상대습도의 값이 크므로 작게 스케일링)
xy_train.iloc[:,0] = xy_train.iloc[:, 0]  % 10000 / 100
xy_train.iloc[:,1] = xy_train.iloc[:, 1]  / 10
xy_train.iloc[:, 6] = xy_train.iloc[:, 6]  / 10
xy_train
# 데이터 파싱
x_data = xy_train.iloc[:,:-1]
y_data = xy_train.iloc[:,-1]

# numpy 형으로 변환
x_train = np.array(x_data)
y_train = np.array(y_data)

# tensor 형으로 변환
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
# W, b값에 대한 학습
W = torch.zeros((7, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr = 1e-5, momentum=0.9)

epochs = 50000

for epoch in range(epochs + 1):
    
  hypothesis = x_train.matmul(W) + b

  cost = torch.mean((y_train - hypothesis)**2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 5000 == 0:
    print("{} {:.2f}".format(epoch, cost.item()))
# 테스트 데이터 받아오기
xy_test = pd.read_csv('../input/2020-ai-term-project-18011817/test_seoul_grandpark.csv', header=None, skiprows=1)

# 데이터 프로세싱 (train 데이터와 동일하게 전처리)
xy_test.iloc[:, 0] = xy_test.iloc[:, 0]  % 10000 / 100
xy_test.iloc[:, 1] = xy_test.iloc[:, 1]  / 10
xy_test.iloc[:, 6] = xy_test.iloc[:, 6]  / 10

# 테스트 데이터를 tensor형으로 변환
x_test = torch.FloatTensor(np.array(xy_test))

prediction = x_test.matmul(W) + b
# 제출 형식 데이터 받아오기
submit = pd.read_csv('../input/2020-ai-term-project-18011817/submit_sample.csv')

for i in range(len(prediction)):
  submit["Expected"][i] = prediction[i].item()

submit.to_csv('baseline.csv', mode='w', header=True, index = False)