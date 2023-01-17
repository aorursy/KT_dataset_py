!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c solarenergy-meteorologicalphenomenon2
!unzip solarenergy-meteorologicalphenomenon2.zip
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
learning_rate = 1e-5
training_epochs = 700
batch_size = 50

#Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('Solar_TrainData_3.csv',header=None,skiprows=[0], usecols=range(1,9))
test_data=pd.read_csv('Solar_TestData_2.csv',header=None,skiprows=[0], usecols=range(1,8))
x_train_data=train_data.loc[:,0:7]
y_train_data=train_data[[8]]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
#x_train_data = Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
print(x_train_data)
print(y_train_data)

train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(7,1024,bias=True)
linear2 = torch.nn.Linear(1024,512,bias=True)
linear3 = torch.nn.Linear(512,256,bias=True)
linear4 = torch.nn.Linear(256,128,bias=True)
linear5= torch.nn.Linear(128,64,bias=True)
linear6= torch.nn.Linear(64,32,bias=True)
linear7= torch.nn.Linear(32,1,bias=True)# layer 올리고 은닉층 숫자 넣어주기
relu= torch.nn.ReLU()#relu 활성화 함수 사용
# Random Init => Xavier Init

torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(linear4.weight)

torch.nn.init.xavier_uniform_(linear5.weight)

torch.nn.init.xavier_uniform_(linear6.weight)

torch.nn.init.xavier_uniform_(linear7.weight)


# ======================================
# relu는 맨 마지막 레이어에서 빼는 것이 좋다.
# ======================================
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4,relu,
                            linear5,relu,
                            linear6,relu,
                            linear7
                            ).to(device)
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
  #x_test_data = Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('Solar_SubmitForm_2.csv')
submit
test1_date=pd.read_csv('Solar_TestData_2.csv',header=None,skiprows = [0])
test1_data=test1_date.loc[:,0]
test1_data=np.array(test1_data)

print(test1_data)
for i in range(len(correct_prediction)):
  submit['YYYY/MM/DD'][i]=test1_data[i]
  submit['Predict'][i]=correct_prediction[i].item()

submit#날짜 넣어주기

submit.to_csv('submit.csv',index=False,header=True)

!kaggle competitions submit -c solarenergy-meteorologicalphenomenon2 -f submit.csv -m "Message"
