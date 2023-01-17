import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler  # 데이터 정규화
import random
import matplotlib.pyplot as plt
device = torch.device('cuda') # 디바이스 GPU 설정
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)

learning_rate = 1e-4
training_epochs = 1000
batch_size = 15
#drop_prob = 0.3
train = pd.read_csv('train.csv', header=None, skiprows=1)
test = pd.read_csv('test.csv', header=None, skiprows=1)
train[0] = train[0]%10000/100
x_train = train.loc[:,0:9]
y_train = train.loc[:,[10]]

x_train = np.array(x_train)
y_train = np.array(y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
dataset = TensorDataset(x_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
linear1 = torch.nn.Linear(10,512,bias=True)
linear2 = torch.nn.Linear(512,512,bias=True)
linear3 = torch.nn.Linear(512,1,bias=True)
leakyrelu = torch.nn.LeakyReLU()

torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)

model = torch.nn.Sequential(linear1,leakyrelu,
                            linear2,leakyrelu,
                            linear3
                            ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss = torch.nn.MSELoss().to(device)

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
    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.4f}'.format(avg_cost))
  losses.append(cost.item())
print('Learning finished')
print(min(err_history))
plt.plot(losses)
plt.plot(err_history)
plt.show()
best_model = model_history[np.argmin(err_history)]
min(err_history)
test[0] = test[0]%10000/100
x_test = test.loc[:,:]
x_test = np.array(x_test)
x_test = scaler.transform(x_test)
x_test = torch.from_numpy(x_test).float().to(device)

with torch.no_grad():
    #model.eval()     
    predict = best_model(x_test)
submit = pd.read_csv('submit_sample.csv')
submit['Total'] = submit['Total'].astype(float)
for i in range(len(predict)):
  submit['Total'][i] = predict[i]
submit.to_csv('submit.csv', mode = 'w', index = False, header = True)
submit