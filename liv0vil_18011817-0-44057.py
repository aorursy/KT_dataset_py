import torch
import pandas as pd
import numpy as np
import random
device = torch.device('cuda') # 디바이스 GPU 설정
torch.manual_seed(777)
random.seed(777)
torch.cuda.manual_seed_all(777)
scaler = preprocessing.StandardScaler()
train = pd.read_csv('..input/predict-numdeath/numDeath_train.csv', header=None, skiprows=1, usecols=range(2,7))
train = train.dropna()
test = pd.read_csv('..input/predict-numdeath/numDeath_test.csv', header = None, skiprows=1,usecols=range(1,5))
test = test.dropna()
xdata = train.loc[0:,1:5]
ydata = train.loc[:,6]

x_train_data = np.array(xdata)
y_train_data = np.array(ydata)

x_train = torch.FloatTensor(x_train_data).to(device)
y_train = torch.FloatTensor(y_train_data).to(device)

test_data = test.loc[0:,1:4] 
x_test = np.array(test_data)
x_test = torch.FloatTensor(x_test).to(device)
linear1 = torch.nn.Linear(4, 1)
torch.nn.init.xavier_normal_(linear1.weight)
model = torch.nn.Sequential(linear1).to(device)
loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6)

nb_epochs = 105000

for epoch in range(nb_epochs + 1):

  hypothesis = model(x_train)
  cost = loss(hypothesis, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()


  if epoch % 100== 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
prediction = model(x_test)
submit = pd.read_csv('..input/predict-numdeath/submit.csv')

for i in range(len(prediction)) :
  submit['Expected'][i] = prediction[i].item()
submit.to_csv('submit.csv' , mode = 'w' , index = False)
submit
!kaggle competitions submit -c predict-numdeath -f submit.csv -m "18011817 홍주영"