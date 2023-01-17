import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/insurance/insurance.csv')
for feature in list(data):
    print(data[feature].isnull().values.any()) # check null
categorical_cols = list(data.select_dtypes(include='object').columns)
print(categorical_cols)
for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
print(data)
import torch
train, test = train_test_split(data, test_size = 0.2, random_state = 42)
x_train = train.drop(['charges'], axis=1)
x_train = pd.get_dummies(x_train)
y_train = train['charges']

x_train = torch.FloatTensor(x_train.values)
y_train = torch.FloatTensor(y_train.values)
print(x_train)
y_train = y_train.view([-1,1])
print(y_train)
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
n_features = len(x_train[0])
model = nn.Linear(n_features,1)
optimizer = optim.SGD(model.parameters(), lr=0.0003)

nb_epochs = 100000
for epoch in range(nb_epochs):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 10000 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
print(list(model.parameters()))
        

x_test = test.drop(['charges'], axis=1)
x_test = pd.get_dummies(x_test)
y_test = test['charges']

x_test = torch.FloatTensor(x_test.values)
y_test = torch.FloatTensor(y_test.values)
y_test = y_test.view([-1,1])
cost = F.mse_loss(model(x_test), y_test) 
print(cost)
