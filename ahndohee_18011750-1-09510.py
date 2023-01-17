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
training_epochs = 163
batch_size = 10
#dropout 추가
drop_prob=0.5
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('train_unemployment_rate.csv',header=None)
test_data=pd.read_csv('test_unemployment_rate.csv',header=None)
#년,월값 수정
train_data[[1]]=train_data[[1]]%10000/100
x_train_data=train_data.loc[1:,1:3]
y_train_data=train_data.loc[1:,4]
x_train_data
x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)

# 스케일러를 통해 preprocessing
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#layer1->layer3
linear1 = torch.nn.Linear(3,3,bias=True)
linear2 = torch.nn.Linear(3,3,bias=True)
linear3 = torch.nn.Linear(3,1,bias=True)
relu=torch.nn.Dropout(p=drop_prob)
dropout = torch.nn.Dropout(p=drop_prob)
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3).to(device)
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
with torch.no_grad():
  test_data[[1]]=test_data[[1]]%10000/100
  x_test_data=test_data.loc[1:,1:3]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
correct_prediction
submit=pd.read_csv('submission.csv')
for i in range(len(correct_prediction)):
  submit['Expected'][i]=float(correct_prediction[i])
  submit['Expected']=submit['Expected'].astype(float)
submit