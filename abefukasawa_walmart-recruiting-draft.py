# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', parse_dates=True)

train.info()
train.head()
train['Date'] = pd.to_datetime(train.Date)
train.info()
len(train)
grouped = train.groupby('Date').Weekly_Sales.sum().reset_index()
type(grouped)
grouped.head()
len(grouped)
from matplotlib import pyplot as plt

%matplotlib inline

plt.figure(figsize=(20, 10))

plt.plot(grouped.Date, grouped.Weekly_Sales)
train.columns
for i in range(1,7):

    grouped["lag_{}".format(i)] = grouped.Weekly_Sales.shift(i)

grouped.head()
grouped.head(10)
plt.figure(figsize=(20, 7))

plt.plot(grouped.Date, grouped.Weekly_Sales)

plt.plot(grouped.Date, grouped.lag_1)

plt.plot(grouped.Date, grouped.lag_2)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()

grouped.set_index('Date')

grouped.fillna(49750740.50, inplace=True)

grouped_train = grouped.iloc[0:120, :].values

grouped_test = grouped.iloc[120:, :].values
X_train = grouped_train[:, 2:]

y_train = grouped_train[:, 1]

rf.fit(X_train, y_train)
X_test = grouped_test[:, 2:]

y_test = grouped_test[:, 1]



y_hat = rf.predict(X_test)
from sklearn.metrics import mean_squared_error as mse
print(mse(y_test, y_hat)**0.5)
plt.scatter(y_test, y_hat)
plt.figure(figsize=(20, 7))

plt.plot(grouped.Date[120:], y_test)

plt.plot(grouped.Date[120:], y_hat)