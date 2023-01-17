! pip uninstall --y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c traffic-accident
! unzip traffic-accident.zip
import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu' #GPU 연결되어 있으면 GPU 쓰고 아니면 CPU쓰기

# 디버깅 편리하게 하기 위해서 seed 고정시키기
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate = 0.02
training_epochs = 63
batch_size = 50
drop_prob = 0.3
Scaler = preprocessing.StandardScaler()
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.drop(['snowFall','deepSnowfall'], axis=1, inplace=True)
test_data.drop(['snowFall','deepSnowfall'], axis=1, inplace=True)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

x_train = train_data.loc[:,'avgTemp':'fogDuration']
y_train = train_data['trafficAccident']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)
linear1 = torch.nn.Linear(6, 4, bias = True)
linear2 = torch.nn.Linear(4, 4, bias = True)
linear3 = torch.nn.Linear(4, 4, bias = True)
linear4 = torch.nn.Linear(4, 2, bias = True)
linear5 = torch.nn.Linear(2, 2, bias = True)
linear6 = torch.nn.Linear(2, 1, bias = True)

torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)
torch.nn.init.kaiming_normal_(linear4.weight)
torch.nn.init.kaiming_normal_(linear5.weight)
torch.nn.init.kaiming_normal_(linear6.weight)
relu= torch.nn.ReLU() #LeakyReLU()
model = torch.nn.Sequential(linear1,
                            linear2,
                            linear3,
                            linear4,
                            linear5,
                            linear6
                            )

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_batch = len(data_loader)
for epoch in range(training_epochs):
  avg_cost = 0
  for X,Y in data_loader:
    X = X
    Y = Y

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis , Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/total_batch
  print('Epoch:', '%04d' %(epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))
with torch.no_grad():
  x_test = test_data.loc[:,'avgTemp':'fogDuration']
  x_test = np.array(x_test)
  #x_test = Scaler.transform(x_test)
  x_test = torch.from_numpy(x_test).float()

  prediction = model(x_test)
  correct_prediction = prediction.cpu().numpy().reshape(-1,1)
    
submit = pd.read_csv('submit_sample.csv')

for i in range(len(correct_prediction)):
  submit['Expected'][i] = correct_prediction[i].item()
submit
submit.to_csv('submission.csv',index=False,header=True)
! kaggle competitions submit -c traffic-accident -f submission.csv -m "Message"