import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from torch.utils.data import  TensorDataset, DataLoader
device = torch.device('cuda') 
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 0.0001
training_epochs = 5000
batch_size = 100
drop_prob = 0.3
train = pd.read_csv('train_wave.csv', header = None, skiprows=1, usecols=range(2, 13))
x_data = train.loc[:1705, 1:11]
y_data = train.loc[:1705, [12]]
x_data = np.array(x_data)
y_data = np.array(y_data)

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(x_data)

x_train = torch.FloatTensor(x_data).to(device)
y_train = torch.FloatTensor(y_data).to(device) 

train_dataset = TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           drop_last = True)
linear1 = torch.nn.Linear(10, 10,bias=True)
linear2 = torch.nn.Linear(10, 1,bias=True)

torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)

relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1,relu,
                            linear2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []

total_batch = len(data_loader)
model.train()

for epoch in range(5000 + 1):
  avg_cost = 0

  for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = torch.mean((hypothesis - Y) ** 2)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
  
  if epoch % 100 == 0:  
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.6f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
xy_test = pd.read_csv('test_wave.csv', header = None, skiprows=1, usecols = range(2, 12))
x_data = xy_test.loc[:, 1:11]
x_data = np.array(x_data)
x_data = scaler.transform(x_data)
x_test = torch.FloatTensor(x_data).to(device)

with torch.no_grad():
    model.eval()
    
    predict = model(x_test)
submit = pd.read_csv('submit_sample.csv')
submit['Expected'] = submit['Expected'].astype(float)
for i in range(len(predict)):
  submit['Expected'][i] = predict[i]
submit.to_csv('submit_sample.csv', mode = 'w', index = False, header = True)