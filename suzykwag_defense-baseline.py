import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

import torchvision.datasets as data
import torchvision.transforms as transforms
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device =='cuda':
  torch.cuda.manual_seed_all(777)

learning_rate = 0.001
batch_size = 100
drop_prob = 0.4
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
train = pd.read_csv('waterquality_train.csv',header=None, skiprows=1)
train[0] = train[0]%10000/100

train = train.values[:,:]
x_train=train[:,:-1]
y_train=train[:,[-1]]

x_train = scaler.fit_transform(x_train)

x_train = torch.FloatTensor(x_train)

y_train = torch.FloatTensor(y_train)

train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(8,512,bias=True)
linear2 = torch.nn.Linear(512,512,bias=True)
linear3 = torch.nn.Linear(512,512,bias=True)
linear4 = torch.nn.Linear(512,512,bias=True)
linear5 = torch.nn.Linear(512,1,bias=True)


relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)


dropout = torch.nn.Dropout(p=drop_prob)

model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3,relu,dropout,
                            linear4,relu,dropout,
                            linear5).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss = torch.nn.MSELoss().to(device)
total_batch = len(data_loader)

for epoch in range(50):
    avg_cost = 0
    model.train()
    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = loss(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(avg_cost))

test = pd.read_csv('waterquality_test.csv',header=None, skiprows=1)

test[0] = test[0]%10000/100
with torch.no_grad():
  model.eval()
  x_test = test.values[:,:]
  x_test = np.array(x_test)
  x_test = scaler.transform(x_test)
  x_test = torch.from_numpy(x_test).float().to(device)
  
  prediction = model(x_test)
prediction = prediction.cpu().numpy().reshape(-1,1)
submit = pd.read_csv('waterquality_submit.csv')

for i in range(len(prediction)):
    submit['Expected'][i] = prediction[i].item()

submit['Expected']=submit['Expected'].astype(int)

submit.to_csv('waterquality_submit.csv',index=False)