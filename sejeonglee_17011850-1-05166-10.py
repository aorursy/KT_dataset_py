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
learning_rate = 0.01
training_epochs = 200
batch_size = 1
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('train-horse-win-rate.csv',header=None,skiprows=1,usecols=range(1,8))
test_data=pd.read_csv('test-horse-win-rate.csv',header=None,skiprows=1,usecols=range(1,7))
x_train_data=train_data.loc[:,0:6]
y_train_data=train_data.loc[:,7]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
linear1 = torch.nn.Linear(6,6,bias=True)
linear2 = torch.nn.Linear(6,1,bias=True)

relu = torch.nn.ELU()
torch.nn.init.kaiming_uniform_(linear1.weight)
torch.nn.init.kaiming_uniform_(linear2.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)
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

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
  print('Learning finished')
# Test the model using test sets
with torch.no_grad():

  x_test_data=test_data.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
  #correct_prediction = torch.argmax(prediction, 1)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)

submit=pd.read_csv('submit_sample.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()

submit
submit.to_csv('submit.csv',index=False,header=True)

!kaggle competitions submit -c horse-win-rate -f submit.csv -m "이세정"