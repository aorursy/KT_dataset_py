from sklearn.preprocessing import RobustScaler
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import random
torch.manual_seed(1)
from dateutil.parser import parse
import time
train_csv = pd.read_csv('/content/seaice_train.csv')
test_csv = pd.read_csv('./seaice_test.csv')
train_csv.head()
train_csv.info()
test_csv.info()
from datetime import datetime as dt
from dateutil.parser import parse
import time
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    # 초단위로 1년 / 현재날짜 의 비율 
    fraction = yearElapsed/yearDuration
    return date.year + fraction
train_decimal_year = []

train_decimal_year.append(toYearFraction(parse("1978-11-15")))
train_decimal_year.append(toYearFraction(parse("1978-12-15")))

for i in range(1979, 2019):
  if i == 2016:
    continue
  for m in range(1 , 13):
    flag = int(m / 10)
    if flag == 0:
      date = ("{}-0{}-15".format(i,m))
    else:
      date = ("{}-{}-15".format(i,m))
    train_decimal_year.append(toYearFraction(parse(date)))

train_decimal_year.append(toYearFraction(parse("2019-01-15")))
train_decimal_year.append(toYearFraction(parse("2019-02-15")))
train_decimal_year.append(toYearFraction(parse("2019-03-15")))
train_decimal_year.append(toYearFraction(parse("2019-04-15")))
train_decimal_year.append(toYearFraction(parse("2019-05-15")))
train_decimal_year = np.array(train_decimal_year)
train_decimal_year.shape
train_csv = train_csv.drop(['month'],axis = 1)

for i in range(475):
  train_csv.iloc[i,0] = train_decimal_year[i]
train_csv.head()
x_carbon = np.array(train_csv.iloc[0:447,0:4])
x_seaice = np.array(train_csv.iloc[0:447,5])
y_carbon = np.array(train_csv.iloc[0:447,4])
scaler = RobustScaler()
x_carbon_s = scaler.fit_transform(x_carbon)
x_ctrain = torch.FloatTensor(x_carbon_s)
y_ctrain = torch.FloatTensor(np.transpose(y_carbon[np.newaxis]))
W = torch.zeros((4,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
import torch.nn.functional as F
optimizer = optim.Adam([W,b], lr = 0.1)
loss = torch.nn.MSELoss()

nb_epochs = 100000
for epochs in range(nb_epochs + 1):
  h = x_ctrain.matmul(W)+ b
  cost = loss(h,y_ctrain)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epochs % 1000 == 0:
    print("epochs {}/{:4d}, cost {:.6f}".format(epochs, nb_epochs, cost.item()))
carbon_x_test = np.array(train_csv.iloc[447:475,0:4])
carbon_xs_test = scaler.transform(carbon_x_test)
carbon_xs_test
x_ctest = torch.FloatTensor(carbon_xs_test)
carbon_pred = x_ctest.matmul(W)+b
carbon_pred = carbon_pred.detach().numpy()
index = 0
for i in range(447,475):
   train_csv.iloc[i,4] = carbon_pred[index]
   index = index + 1
train_csv
train_csv.info()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1)
torch.manual_seed(1)
if device == 'cuda':
  torch.cuda.manual_seed_all(1)
train_D = np.array(train_csv.iloc[:, 0:5])
train_D = scaler.fit_transform(train_D)
train_L = np.transpose(np.array(train_csv.iloc[:,5])[np.newaxis])
train_D = torch.FloatTensor(train_D)
train_L = torch.FloatTensor(train_L)
learning_rate = 0.01
batch_size= 10
drop_prob = 0.1
dataset = torch.utils.data.TensorDataset(train_D, train_L)
data_loader = torch.utils.data.DataLoader(dataset = dataset,batch_size = batch_size, shuffle= True, drop_last = True)
linear1 = torch.nn.Linear(train_D.shape[1],32,bias=True)
linear2 = torch.nn.Linear(32,32,bias=True)
linear3 = torch.nn.Linear(32,32,bias=True)
linear4 = torch.nn.Linear(32,1,bias=True) 
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=drop_prob)
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4).to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
total_batch = len(data_loader)
for epoch in range(1001):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.view(-1, train_D.shape[1]).to(device)      
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
    if epoch % 10 == 0:
      print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
test_decimal_year = []

for m in range(1 , 13):
  flag = int(m / 10)
  if flag == 0:
    date = ("2016-0{}-15".format(m))
  else:
    date = ("2016-{}-15".format(m))
  test_decimal_year.append(toYearFraction(parse(date)))

test_decimal_year = np.array(test_decimal_year )
test_decimal_year
test_csv = test_csv.drop(['month'],axis = 1)

for i in range(12):
  test_csv.iloc[i,0] = test_decimal_year[i]
test_D = scaler.transform(np.array(test_csv))

test_D = torch.FloatTensor(test_D)
test_D
model.eval()
pred = model(test_D.to(device))
while torch.no_grad():
  model.eval()
  pred=  model(test_D.to(device))
sample = pd.read_csv('./seaice_sample.csv')
sample['seaice_extent'] = pred.cpu().detach().numpy()
sample.to_csv('ice.csv', index =False)