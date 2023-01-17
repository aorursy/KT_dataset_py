! pip uninstall --y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6

! mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle -v
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
! kaggle competitions download -c 2020soil
! unzip 2020soil.zip
learning_rate = 1e-4
training_epoches = 100
batch_size = 50
Scaler = preprocessing.StandardScaler() #MinMax도 참고하기

drop_prob=0.3
dropout = torch.nn.Dropout(p=drop_prob)
train = pd.read_csv('2020AI_soil_train.csv', header=None, skiprows=1,usecols=range(1,9))

train
test = pd.read_csv('2020_soil_test.csv', header = None, skiprows=1, usecols=range(1,8))

test
x_train = train.loc[:,1:7]
y_train = train.loc[:,8:8]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last=True)
linear1 = torch.nn.Linear(7,64, bias = True) # feature
linear2 = torch.nn.Linear(64,64, bias = True)
linear3 = torch.nn.Linear(64,64, bias = True)
linear4 = torch.nn.Linear(64,32, bias = True)
linear5 = torch.nn.Linear(32,32, bias = True)
linear6 = torch.nn.Linear(32,32, bias = True)
linear7 = torch.nn.Linear(32,16, bias = True)
linear8 = torch.nn.Linear(16,16, bias = True)
linear9 = torch.nn.Linear(16,8, bias = True)
linear10 = torch.nn.Linear(8,8, bias = True)
linear11 = torch.nn.Linear(8,4, bias = True)
linear12 = torch.nn.Linear(4,1, bias = True)

relu = torch.nn.LeakyReLU()
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
torch.nn.init.xavier_uniform_(linear11.weight)
torch.nn.init.xavier_uniform_(linear12.weight)
model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4,relu,
                            linear5,relu,
                            linear6,relu,
                            linear7,relu,
                            linear8,relu,
                            linear9,relu,
                            linear10,relu,
                            linear11,relu,
                            linear12).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epoches):
  avg_cost = 0

  for X, Y in data_loader:

    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
  
  print('Epoch:','%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
print('Learning finshed')
with torch.no_grad():
  x_test = test.loc[:,1:7]
  x_test = np.array(x_test)

  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('soil_submission.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i] = correct_prediction[i].item()

submit
submit.to_csv('submission.csv', mode='w', index = False)
! kaggle competitions submit -c 2020soil -f submission.csv -m "기말고사"