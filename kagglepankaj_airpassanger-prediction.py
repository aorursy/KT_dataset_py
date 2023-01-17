# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data loading from kaggle

data=pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

data.head()
data.Month=pd.to_datetime(data.Month)

data.set_index('Month',inplace=True)

data.plot()
#trying make stationary by rolling mean by tacking log values

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

logged=np.log(data)

plt.plot(logged,label='original')

plt.plot(logged.rolling(12).mean(),label='rolling mean')

plt.plot(logged.rolling(12).std(),label='rolling std')

plt.legend()
#loged values followed by one step lag difference

logged_diff=logged.diff().dropna()

plt.figure(figsize=(10,5))

plt.plot(logged_diff,label='original')

plt.plot(logged_diff.rolling(12).mean(),label='rolling mean')

plt.plot(logged_diff.rolling(12).std(),label='rolling std')

plt.legend()
from statsmodels.tsa.stattools import adfuller

adfuller(logged_diff)
# 2nd value goes 0 so choose q=2



from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(logged_diff,lags=10);
# Similar as previous one choose p=2

plot_pacf(logged_diff,lags=10);
from statsmodels.tsa.statespace.sarimax import SARIMAX

model=SARIMAX(logged,order=(2,1,2),seasonal=True,seasonal_order=(2,1,2,12)).fit(disp=-1)
plt.figure(figsize=(10,10))

plt.plot(logged)

plt.plot(model.fittedvalues)

np.abs(model.resid).mean()
(model.fittedvalues.cumsum()+logged.iloc[0].values)
plt.figure(figsize=(10,10))

plt.plot(data)

plt.plot(np.exp(model.fittedvalues))
model.summary()
y_pred=model.predict(start='1960-12',end='1970-10',dynamic=True)

forecasted=np.exp(y_pred)
plt.figure(figsize=(10,10))

plt.plot(data)

plt.plot(forecasted)