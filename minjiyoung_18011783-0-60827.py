!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c 2020-ai-term-project-18011759
!unzip 2020-ai-term-project-18011759.zip
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
training_epochs = 300
batch_size = 200
train=pd.read_csv('train.CSV', encoding='euc-kr',header=None, skiprows=1, usecols=range(0,8))

train[0] = train[0]%10000/100

data_x = train.loc[:, 2:6]
data_y = train.loc[:,[7]]

data_x = np.array(data_x)
data_y = np.array(data_y)

# 스케일러 활용
#Scaler = preprocessing.StandardScaler()
#data_x = Scaler.fit_transform(data_x)

x_train = torch.FloatTensor(data_x)
y_train = torch.FloatTensor(data_y)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(5,32, bias = True)
linear2 = torch.nn.Linear(32,32, bias = True)
linear3 = torch.nn.Linear(32,32, bias = True)
linear4 = torch.nn.Linear(32,32, bias = True)
linear5 = torch.nn.Linear(32,32, bias = True)
linear6 = torch.nn.Linear(32,32, bias = True)
linear7 = torch.nn.Linear(32,1, bias = True)

relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)
model = torch.nn.Sequential(linear1, relu,
                            linear2, relu,
                            linear3, relu, 
                            linear4, relu,
                            linear5, relu, 
                            linear6, relu,
                            linear7).to(device)
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
test_xy = pd.read_csv("test.CSV", encoding='euc-kr',header=None, skiprows=1, usecols=range(0,7))
test_xy[0] = test_xy[0]%10000/100

with torch.no_grad():

  x_test_data=test_xy.loc[:,2:6]
  x_test_data=np.array(x_test_data)
 # x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('submit_sample.CSV')
for i in range(len(correct_prediction)):
  submit['total'][i]=correct_prediction[i].item()
submit.to_csv('submit.csv',index=False,header=True)
!kaggle competitions submit -c 2020-ai-term-project-18011759 -f submit.csv -m "Message"