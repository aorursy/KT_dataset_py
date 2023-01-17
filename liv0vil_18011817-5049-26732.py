import pandas as pd

import numpy as np



import torch

import torch.optim as optim

import torchvision.datasets as data

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader



from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화

import random

import matplotlib.pyplot as plt
torch.manual_seed(777)

random.seed(777)

torch.cuda.manual_seed_all(777)



learning_rate = 0.1

training_epochs = 30

batch_size = 5
train_data=pd.read_csv('../input/sejongyjelectricpowerprediction/electric_power_train_data.csv')

test_data=pd.read_csv('../input/sejongyjelectricpowerprediction/electric_power_test_data.csv')
train_data['Date'] = train_data['Date']%1000000/10000

test_data['Date'] = test_data['Date']%1000000/10000



x_train=train_data.loc[:,[i for i in train_data.keys()[:-1]]]

y_train=train_data[train_data.keys()[-1]]



x_train=np.array(x_train)

y_train=np.array(y_train)



scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)



x_train=torch.FloatTensor(x_train)

y_train=torch.FloatTensor(y_train)
dataset = TensorDataset(x_train, y_train)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
linear1 = torch.nn.Linear(3,1,bias=True)

linear2 = torch.nn.Linear(1,1,bias=True)

leakyrelu = torch.nn.LeakyReLU()



torch.nn.init.xavier_normal_(linear1.weight)

torch.nn.init.xavier_normal_(linear2.weight)





model = torch.nn.Sequential(linear1,leakyrelu,linear2)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

loss = torch.nn.MSELoss()



losses = []

model_history = []

err_history = []



total_batch = len(data_loader)



for epoch in range(training_epochs + 1):

  avg_cost = 0

  model.train()

  

  for X, Y in data_loader:

    X = X

    Y = Y



    optimizer.zero_grad()

    hypothesis = model(X)

    cost = loss(hypothesis, Y)

    cost.backward()

    optimizer.step()



    avg_cost += cost / total_batch

    

  model_history.append(model)

  err_history.append(avg_cost)

  

  if epoch % 5 == 0:  

    print('Epoch:', '%d' % (epoch + 1), 'Cost =', '{:.4f}'.format(avg_cost))

  losses.append(cost.item())

print('Learning finished')
print(min(err_history))
plt.plot(losses)

plt.plot(err_history)

plt.show()
best_model = model_history[np.argmin(err_history)]

min(err_history)
with torch.no_grad():



  x_test_data=test_data.loc[:,[i for i in test_data.keys()[:]]]

  x_test_data=np.array(x_test_data)

  x_test_data = scaler.transform(x_test_data)

  x_test_data=torch.from_numpy(x_test_data).float()



  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)

submit=pd.read_csv('../input/sejongyjelectricpowerprediction/electric_power_submit_data.csv')

for i in range(len(correct_prediction)):

  submit['ElectricPower'][i]=correct_prediction[i].item()

submit.to_csv('submit.csv', mode = 'w', index = False, header = True)

submit