! pip uninstall -y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c ai-project-life-environment
!unzip ai-project-life-environment.zip
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
learning_rate = 0.2
training_epochs = 100
batch_size = 10
train=pd.read_csv('train data.csv',header=None, skiprows=1)

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

#레이어 깊게 쌓음
linear1 = torch.nn.Linear(8,5, bias = True)
linear2 = torch.nn.Linear(5,5, bias = True)
linear3 = torch.nn.Linear(5,5, bias = True)
linear4 = torch.nn.Linear(5,5, bias = True)
linear5 = torch.nn.Linear(5,5, bias = True)
linear6 = torch.nn.Linear(5,5, bias = True)
linear7 = torch.nn.Linear(5,5, bias = True)
linear8 = torch.nn.Linear(5,5, bias = True)
linear9 = torch.nn.Linear(5,5, bias = True)
linear10 = torch.nn.Linear(5,1, bias = True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)

#xavier_uniform 사용
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)
torch.nn.init.xavier_uniform_(linear8.weight)
torch.nn.init.xavier_uniform_(linear9.weight)
torch.nn.init.xavier_uniform_(linear10.weight)


#relu사용
model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5, relu, dropout,
                            linear6, relu, dropout,
                            linear7, relu, dropout,
                            linear8, relu, dropout,
                            linear9, relu, dropout,
                            linear10).to(device)
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
test_xy = pd.read_csv("test data.csv", header=None, skiprows=1)
test_xy[0] = test_xy[0]%10000/100

with torch.no_grad():

  x_test_data=test_xy.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submit sample.csv')
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()
submit.to_csv('submit.csv',index=False,header=True)