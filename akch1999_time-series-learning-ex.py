# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('../input/temperature.csv', index_col = 'datetime', parse_dates=['datetime'])

df.head()
df = df['Los Angeles']
df = pd.DataFrame(df)
df.head()
df = df.rename(columns={'Los Angeles': "Temp"})
df.head()
df.plot()
df['l3'] = df['Temp'].shift(3)

df['l2'] = df['Temp'].shift(2)

df['l1'] = df['Temp'].shift(1)
df.head()
df = df.dropna()
df.head()
X = df[['l1', 'l2', 'l3']]

Y = df['Temp'] # 45244
x_train = X.iloc[:20000]

y_train = Y.iloc[:20000]

x_test = X.iloc[20000:]

y_test = Y.iloc[20000:]
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
forecast = regr.predict(x_test)
mean_squared_error(forecast, y_test)
r2_score(forecast, y_test)
y_train.asfreq('W').plot()

y_test.asfreq('W').plot()
forecast
f = pd.DataFrame({'Temp': forecast}, index = y_test.index)

f.head()
f.asfreq('W').plot()

y_test.asfreq('W').plot()

y_train.asfreq('W').plot()

plt.legend(['Forecast', 'Actual', 'Trained Data'])