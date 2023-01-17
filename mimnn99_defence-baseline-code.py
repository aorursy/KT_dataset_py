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
learning_rate = 1e-4
training_epoches = 700
batch_size = 50
Scaler = preprocessing.StandardScaler()
# 저는 제가 만든 csv파일 바로 넣어서 했는데, 여러분은 ! kaggle 해서 다운로드 받아서 하시면 됩니다!
# ! kaggle competitions download -c solarenergy-meteorologicalphenomenon2
# ! unzip solarenergy-meteorologicalphenomenon2.zip
train = pd.read_csv('Solar_TrainData_3.csv', header=None, skiprows=1, usecols=range(0,9))
train = train.dropna()
train
test = pd.read_csv('Solar_TestData_2.csv', header = None, skiprows=1,usecols=range(0,8))

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
linear1 = torch.nn.Linear(7,32, bias = True) # feature
linear2 = torch.nn.Linear(32,32, bias = True)
linear3 = torch.nn.Linear(32,32, bias = True)
linear4 = torch.nn.Linear(32,16, bias = True)
linear5 = torch.nn.Linear(16,16, bias = True)
linear6 = torch.nn.Linear(16,16, bias = True)
linear7 = torch.nn.Linear(16,8, bias = True)
linear8 = torch.nn.Linear(8,8, bias = True)
linear9 = torch.nn.Linear(8,8, bias = True)
linear10 = torch.nn.Linear(8,1, bias = True)
# layer 5 -> 7- > 10
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
model = torch.nn.Sequential(linear1,
                            linear2,
                            linear3,
                            linear4,
                            linear5,
                            linear6,
                            linear7,
                            linear8,
                            linear9,
                            linear10).to(device)
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
MAKE = pd.read_csv('Solar_TestData_2.csv', header = None, skiprows= 1) 
submit = pd.read_csv('Solar_SubmitForm_2.csv')
submit
for i in range(len(correct_prediction)):
  submit['Predict'][i] = correct_prediction[i].item()

submit['YYYY/MM/DD'] = MAKE[0]
submit
submit.to_csv('Baseline_By_NN.csv', mode='w', index = False)
# 제출 : ) 
# ! kaggle coㅋmpetitions submit -c solarenergy-meteorologicalphenomenon2 -f submission.csv -m "Message"