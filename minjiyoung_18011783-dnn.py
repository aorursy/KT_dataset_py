!pip uninstall kaggle
!pip install --upgrade pipg
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c 2020termproject-18011826
!unzip 2020termproject-18011826.zip
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
learning_rate = 1
training_epochs = 1000
batch_size = 50
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('train_sweetpotato_price.csv',header=None, skiprows=1, usecols=range(1,6))
test_data=pd.read_csv('test_sweetpotato_price.csv',header=None, skiprows=1, usecols=range(1,5))

x_train_data=train_data.loc[:,0:4]
y_train_data=train_data.loc[:,5]

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
linear1 = torch.nn.Linear(4,1,bias=True)
# Random Init => Xavier Init
torch.nn.init.xavier_uniform_(linear1.weight)
# ======================================
# relu는 맨 마지막 레이어에서 빼는 것이 좋다.
# ======================================
model = torch.nn.Sequential(linear1).to(device)
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
# test_data=pd.read_csv('test_sweetpotato_price.csv',header=None, skiprows=1, usecols=range(1,5))
with torch.no_grad():

  x_test_data=test_data.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.fit_transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submit_sample.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()

submit
submit.to_csv('submit.csv',mode='w',index=False)
! kaggle competitions submit -c 2020termproject-18011826 -f submit.csv -m "submit"
