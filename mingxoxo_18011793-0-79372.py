#!kaggle competitions download -c 2020-ai-term-project-18011759
#!unzip 2020-ai-term-project-18011759.zip
import torch

import torchvision.datasets as data

import torchvision.transforms as transforms

import random



import pandas as pd

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'



random.seed(777)

torch.manual_seed(777)

if device == 'cuda':

  torch.cuda.manual_seed_all(777)
learning_rate = 0.001

training_epochs = 200

batch_size = 100
train_data = pd.read_csv('../input/2020-ai-term-project-18011759/train.CSV', encoding = "euc-kr",header = None, skiprows=1, usecols=range(0,8))

test_data = pd.read_csv('../input/2020-ai-term-project-18011759/test.CSV', encoding = "euc-kr", header = None, skiprows=1, usecols=range(0,7))
test_data
train_data
x_train_data = train_data.loc[:, 2:6]

y_train_data = train_data.loc[:, [7]]
  

x_train_data = np.array(x_train_data)

y_train_data = np.array(y_train_data)

x_train_data = torch.FloatTensor(x_train_data)

y_train_data = torch.FloatTensor(y_train_data)
train_dataset =  torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 

                                          batch_size=batch_size, 

                                          shuffle = True, 

                                          drop_last = True)
linear1 = torch.nn.Linear(5,530, bias = True)

linear2 = torch.nn.Linear(530,530, bias = True)

linear3 = torch.nn.Linear(530,530, bias = True)

linear4 = torch.nn.Linear(530,530, bias = True)

linear5 = torch.nn.Linear(530,530, bias = True)

linear6 = torch.nn.Linear(530,530, bias = True)

linear7 = torch.nn.Linear(530,1, bias = True)

ReLU = torch.nn.SELU()

dropout = torch.nn.Dropout(p=0.3)
torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

torch.nn.init.xavier_uniform_(linear3.weight)

torch.nn.init.xavier_uniform_(linear4.weight)

torch.nn.init.xavier_uniform_(linear5.weight)

torch.nn.init.xavier_uniform_(linear6.weight)

torch.nn.init.xavier_uniform_(linear7.weight)
model = torch.nn.Sequential(linear1, ReLU,dropout,

                            linear2, ReLU,dropout,

                            linear3, ReLU,dropout,

                            linear4, ReLU,dropout,

                            linear5, ReLU,dropout,

                            linear6, ReLU,dropout,

                            linear7).to(device)
loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,eps=1e-20)

sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
total_batch = len(data_loader)

for epoch in range(training_epochs):

  avg_cost = 0



  for X, Y in data_loader:

    X = X.to(device)

    Y = Y.to(device)



    optimizer.zero_grad()

    hypothesis = model(X)

    cost = loss(hypothesis, Y)

    cost.backward()

    optimizer.step()



    avg_cost += cost/total_batch

  print(epoch+1)

  print(avg_cost)

  if avg_cost < 0.59 :

    break

print("f")
with torch.no_grad():

  x_test = test_data.loc[:,2:6]

  x_test = np.array(x_test)

  x_test = torch.from_numpy(x_test).float().to(device)

  prediction = model(x_test)
correct = prediction.cpu().numpy().reshape(-1, 1)
submit = pd.read_csv('../input/2020-ai-term-project-18011759/submit_sample.CSV')

submit
for i in range(len(correct)):

  submit['total'][i] = correct[i].item()

submit
submit.to_csv('submit.csv', index = False, header = True)
#!kaggle competitions submit -c 2020-ai-term-project-18011759 -f submit.csv -m "result"