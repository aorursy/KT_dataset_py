from sklearn.preprocessing import RobustScaler
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import random
torch.manual_seed(1)
seaice_data = pd.read_csv('../input/prediction-of-sea-ice/seaice_train.csv')
seaice_data
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
seaice_data = seaice_data.drop(['month'],axis = 1)
for i in range(475):
  seaice_data.iloc[i,0] = train_decimal_year[i]
seaice_data.tail(30)
seaice_data.iloc[447:475]
x_carbon = np.array(seaice_data.iloc[0:447,0:4])
x_seaice = np.array(seaice_data.iloc[0:447,5])
y_carbon = np.array(seaice_data.iloc[0:447,4])
x_carbon
scaler = RobustScaler()
x_carbon_s = scaler.fit_transform(x_carbon)
x_carbon_s
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
carbon_x_test = np.array(seaice_data.iloc[447:475,0:4])
carbon_xs_test = scaler.fit_transform(carbon_x_test)
carbon_xs_test
x_ctest = torch.FloatTensor(carbon_xs_test)
carbon_pred = x_ctest.matmul(W)+b
carbon_pred = carbon_pred.detach().numpy()
carbon_pred
index = 0
for i in range(447,475):
   seaice_data.iloc[i,4] = carbon_pred[index]
   index = index + 1
seaice_data
x_data = np.array(seaice_data.iloc[:,0:5])
xs_data = scaler.fit_transform(x_data)
y_data = np.transpose(np.array(seaice_data.iloc[:,5])[np.newaxis])
x_train = torch.FloatTensor(xs_data)
y_train = torch.FloatTensor(y_data)
x_W = torch.zeros((5,1), requires_grad=True)
x_b = torch.zeros(1, requires_grad=True)
optimizer_1 = optim.Adam([x_W,x_b], lr = 1e-3)
loss = torch.nn.MSELoss()

nb_epochs = 50000
for epochs in range(nb_epochs + 1):
  h_1 = x_train.matmul(x_W)+ x_b

  cost_1 = loss(h_1,y_train)
  optimizer_1.zero_grad()
  cost_1.backward()
  optimizer_1.step()

  if epochs % 1000 == 0:
    print("epochs {}/{:4d}, cost {:.6f}".format(epochs, nb_epochs, cost_1.item()))
test_data = pd.read_csv('../input/prediction-of-sea-ice/seaice_test.csv')
test_data
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
test_data = test_data.drop(['month'],axis = 1)

for i in range(12):
  test_data.iloc[i,0] = test_decimal_year[i]
tests_data = scaler.fit_transform(np.array(test_data))

tests_data = torch.FloatTensor(tests_data)
print(tests_data)
predict_h = tests_data.matmul(x_W) + x_b
predict_h
month_ = np.array(range(12)) +1
month_ = np.transpose(month_[np.newaxis])
predict_h = predict_h.detach().numpy()
print(month_.shape)
print(predict_h.shape)
sol_ = np.hstack([month_,predict_h])
sol_