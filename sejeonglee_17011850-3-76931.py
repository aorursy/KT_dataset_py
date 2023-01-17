! mkdir -p ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json
! pip uninstall --y kaggle

! pip install --upgrade

! pip install kaggle==1.5.6
!kaggle competitions download -c predict-seoul-house-price

!unzip predict-seoul-house-price.zip
import numpy as np

import torch

import torch.optim as optim

import pandas as pd

import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화

import torchvision.datasets as data

import torchvision.transforms as transforms

import random

from torch.utils.data import  TensorDataset, DataLoader

import matplotlib.pyplot as plt
xy_train = pd.read_csv('train_data.csv', header = None, skiprows=1, usecols=range(4, 8))

x_data = xy_train.loc[ : , 4:6]

y_data = xy_train.loc[ : , [7]]

x_data = np.array(x_data)

y_data = np.array(y_data)



scaler = MinMaxScaler()

x_data = scaler.fit_transform(x_data)



x_train = torch.FloatTensor(x_data).to(device)

y_train = torch.FloatTensor(y_data).to(device) 
train_dataset = TensorDataset(x_train, y_train)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,

                                           batch_size = batch_size, 

                                           shuffle = True, 

                                           drop_last = True)
linear1 = torch.nn.Linear(3, 50,bias=True)

linear2 = torch.nn.Linear(50, 50,bias=True)

linear3 = torch.nn.Linear(50, 1,bias=True)



relu = torch.nn.LeakyReLU()



torch.nn.init.xavier_normal_(linear1.weight)

torch.nn.init.xavier_normal_(linear2.weight)

torch.nn.init.xavier_normal_(linear3.weight)





model = torch.nn.Sequential(linear1,relu,

                            linear2,relu,

                            linear3).to(device)
loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



losses = []

model_history = []

err_history = []



total_batch = len(data_loader)



for epoch in range(training_epochs + 1):

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

    

  model_history.append(model)

  err_history.append(avg_cost)

  

  if epoch % 10 == 0:  

    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))

  losses.append(cost.item())

print('Learning finished')

plt.plot(losses)

plt.plot(err_history)

plt.show()
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test_data.csv', header = None, skiprows=1, usecols = range(4, 7))

x_data = xy_test.loc[:, 4:6]

x_data = np.array(x_data)

x_data = scaler.transform(x_data)

x_test = torch.FloatTensor(x_data).to(device)



with torch.no_grad():

    model.eval()  # 주의사항 (dropout=False)

    

    predict = best_model(x_test)
submit = pd.read_csv('submit_form.csv')

submit['price'] = submit['price'].astype(float)

for i in range(len(predict)):

  submit['price'][i] = predict[i]

submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
!kaggle competitions submit -c predict-seoul-house-price -f submit.csv -m "세정"