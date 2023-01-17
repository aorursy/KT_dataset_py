import pandas as pd

import numpy as np



train = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

train_x = train.loc[:, 'year':'rainFall']

train_y = train['avgPrice']



print(train.shape)

print(train_x.shape)

print(train_y.shape)



train_x['year'] = (train_x['year']%10000)*0.01

train_x
train_y
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(train_x)

train_x_std = sc.transform(train_x)
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 4, weights  = "distance", algorithm= 'kd_tree')

regressor.fit(train_x_std, train_y)
#예측

y_train_pred = regressor.predict(train_x_std)

y_train_pred
test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

test_x = test.loc[:, 'year':'rainFall']

test_x['year'] = (test_x['year']%10000)*0.01

test_x_std = sc.transform(test_x)

y_test_pred = regressor.predict(test_x_std)

y_test_pred
submit = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')

for i in range(len(y_test_pred)):

  submit['Expected'][i] = round(y_test_pred[i])

submit.to_csv("submission.csv", index = False, header = True)