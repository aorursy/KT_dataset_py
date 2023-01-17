import seaborn as sns

import pandas as pd

import numpy as np

RANDOM_SEED = 777
data = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

data.head()
data['year'].head()
data['year'] = data['year'] - 20100000

data['month'] = data['year']%10000

data['year'] = data['year']//10000

data['month'] = data['month']//100 + data['month']%100/30

data.head()
sns.set()

sns.pairplot(data)
x = data.drop('avgPrice', axis=1)

y = data['avgPrice']

print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

print(x_train.shape)

print(x_test.shape)
# 데이터 표준화

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

x_train_std=scale.fit_transform(x_train)

x_test_std=scale.transform(x_test)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')



knn.fit(x_train_std, y_train)

pred = knn.predict(x_test_std)



from sklearn.metrics import mean_squared_error

print(mean_squared_error(pred, y_test)**0.5)
test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

test.head()
test['year'] = test['year'] - 20100000

test['month'] = test['year']%10000

test['year'] = test['year']//10000

test['month'] = test['month']//100 + test['month']%100/30

test.head()
test_std = scale.transform(test)

print(test_std.shape)
pred = knn.predict(test_std)

pred[:5]
submit = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv', index_col=0)

submit.head()
submit['Expected'] = pred

submit.head()
submit.to_csv('submit.csv')
x_train = x_train.drop('year', axis=1)

x_test = x_test.drop('year', axis=1)

x_train.head()
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

x_train_std=scale.fit_transform(x_train)

x_test_std=scale.transform(x_test)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=19, weights='distance')



knn.fit(x_train_std, y_train)

npred = knn.predict(x_test_std)



from sklearn.metrics import mean_squared_error

print(mean_squared_error(npred, y_test)**0.5)
test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

test['year'] = test['year'] - 20100000

test['month'] = test['year']%10000

test['year'] = test['year']//10000

test['month'] = test['month']//100 + test['month']%100/30

test = test.drop('year', axis=1)

test_std = scale.transform(test)

npred = knn.predict(test_std)

npred[:5]
from sklearn.metrics import mean_squared_error

print(mean_squared_error(pred, npred)**0.5)
nsubmit = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv', index_col=0)

nsubmit['Expected'] = npred

nsubmit.head()
nsubmit.to_csv('test.csv')