import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
# 학습 파라미터 설정
learning_rate = 1e-3
training_epochs = 800       #-----epochs 1000->800
batch_size = 80                            
#drop_prob = 0.0                           
Scaler = preprocessing.MinMaxScaler()   
 
# 데이터 로드
xy = pd.read_csv('library_train.csv',header=None)
xy = xy.loc[2:181,3:12]

# object형식 -> float형식
xy = xy.astype(float)

# 모든 변인
x_data = xy.loc[:,3:11]
x_data = np.array(x_data)
y_data = xy.loc[:,[12]]  
   
# 정규화
x_data = Scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(np.array(y_data))

# 테스트 데이터 로드
xy_test = pd.read_csv('library_test.csv',header=None)
xy_test = xy_test.loc[2:73,3:11]
xy_test = xy_test.astype(float)
x_test = xy_test


train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#################class##################
class NN(torch.nn.Module):
  def __init__(self):
    super(NN,self).__init__()

    self.layer1 = torch.nn.Linear(9,9,bias=True)
    self.layer2 = torch.nn.Linear(9,9,bias=True)
    self.layer3 = torch.nn.Linear(9,4,bias=True)
    self.layer4 = torch.nn.Linear(4,4,bias=True)
    self.layer5 = torch.nn.Linear(4,1,bias=True)
    self.relu = torch.nn.ELU()     #---------활성함수 ReLU -> ELU
    #self.dropout = torch.nn.Dropout(p=drop_prob)

    torch.nn.init.kaiming_uniform_(self.layer1.weight)   #--------초기화 방법 xavier_uniform -> kaiming_uniform
    torch.nn.init.kaiming_uniform_(self.layer2.weight)
    torch.nn.init.kaiming_uniform_(self.layer3.weight)
    torch.nn.init.kaiming_uniform_(self.layer4.weight)
    torch.nn.init.kaiming_uniform_(self.layer5.weight)


  def forward(self,x):
    out = self.layer1(x)
    out = self.relu(out)
    #out = self.dropout(out)
    out = self.layer2(out)
    out = self.relu(out)
    #out = self.dropout(out)
    out = self.layer3(out)
    out = self.relu(out)
    #out = self.dropout(out)
    out = self.layer4(out)
    out = self.relu(out)
    #out = self.dropout(out)
    out = self.layer5(out)
    return out

model = NN().to(device)
# 손실함수와 최적화 함수
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
model.train()

#losses = []
#model_history = []
#err_history = []

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

    #model_history.append(model)
    #err_history.append(avg_cost)

    if epoch % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost))
    #losses.append(cost.item())
print('Learning finished')
# 테스트
with torch.no_grad():
  model.eval()
  x_test = np.array(x_test)
  x_test = Scaler.transform(x_test)
  x_test = torch.FloatTensor(x_test).to(device)

  prediction = model(x_test)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)