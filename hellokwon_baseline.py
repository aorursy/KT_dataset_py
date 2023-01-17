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

learning_rate = 1e-2
training_epochs = 140
batch_size = 100
Scaler = preprocessing.StandardScaler()
train = pd.read_csv("train.csv", header = None, skiprows=1)

train[0] = train[0]%10000/100

data_x = train.loc[:, 0:4]
data_y = train.loc[:,[5]]

data_x = np.array(data_x)
data_y = np.array(data_y)

scaler = preprocessing.StandardScaler()
data_x = scaler.fit_transform(data_x)

x_train = torch.FloatTensor(data_x)
y_train = torch.FloatTensor(data_y)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
linear1 = torch.nn.Linear(5,8, bias = True)
linear2 = torch.nn.Linear(8,8, bias = True)
linear3 = torch.nn.Linear(8,8, bias = True)
linear4 = torch.nn.Linear(8,8, bias = True)
linear5 = torch.nn.Linear(8,1, bias = True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)
model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, relu, dropout,
                            linear4, relu, dropout,
                            linear5).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
model.train()
for epoch in range(training_epochs + 1):
    avg_cost = 0
    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # Forward 계산
        hypothesis = model(X)
        # Error 계산
        cost = loss(hypothesis, Y)
        # Backparopagation
        cost.backward()
        # 가중치 갱신
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost / total_batch
    if epoch%10 == 0:
      print('Epoch:', '%04d' % (epoch), 'cost =', '{:.1f}'.format(avg_cost))

test_xy = pd.read_csv("test.csv", header=None, skiprows=1)
test_xy[0] = test_xy[0]%10000/100

with torch.no_grad():
  model.eval() 
  x_test_data=test_xy.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data = scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
correct_prediction = prediction.cpu().numpy().reshape(-1,1)
correct_prediction
submit=pd.read_csv('sample.csv')
submit
for i in range(len(correct_prediction)):
  submit['Expected'][i]=correct_prediction[i].item()

submit
submit.to_csv('submit.csv', mode = 'w', index=False)

