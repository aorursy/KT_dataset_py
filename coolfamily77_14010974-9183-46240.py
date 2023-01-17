import pandas as pd
import numpy as np
import random

import torch
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cuda')

torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.1
training_epochs = 8000
batch_size = 200
drop_prob = 0.3
xy_train = pd.read_csv('train_seoul_grandpark.csv', header = None, skiprows=1, usecols=range(1, 8))

x_data = xy_train.loc[: , 1:6]
y_data = xy_train.loc[: , [7]]

x_data = np.array(x_data)
y_data = np.array(y_data)

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)
linear1 = torch.nn.Linear(6, 4,bias=True)
linear2 = torch.nn.Linear(4, 4,bias=True)
linear3 = torch.nn.Linear(4, 1,bias=True)
#dropout = torch.nn.Dropout(p=drop_prob)
relu = torch.nn.SELU()
torch.nn.init.kaiming_normal_(linear1.weight)
torch.nn.init.kaiming_normal_(linear2.weight)
torch.nn.init.kaiming_normal_(linear3.weight)

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
  #model.train()
  
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
  
  if epoch % 100 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
best_model = model_history[np.argmin(err_history)]
xy_test = pd.read_csv('test_seoul_grandpark.csv', header = None, skiprows=1, usecols = range(1, 7))
x_data = xy_test.loc[:, 1:6]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()     
    predict = best_model(x_test)

submit = pd.read_csv('submit_sample.csv')
submit['Expected'] = submit['Expected'].astype(float)
for i in range(len(predict)):
  submit['Expected'][i] = predict[i]
submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
submit