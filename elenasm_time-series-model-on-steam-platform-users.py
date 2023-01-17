import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('/kaggle/input/steam-concurrent-players/Steam Players.csv')

data.head()
print(data.info())



data['DateTime'] = pd.to_datetime(data['DateTime'])

print(data.dtypes)
data['Year'] = data['DateTime'].dt.year

print(data.head(3))



data['Year'].value_counts()
data[data['Users'] != None]['Year'].value_counts()
df = data[data['Year'].isin([2019,2018,2017,2016,2015])]

df['Year'].value_counts()
ts = df['Users']

plt.figure(figsize=(20,10))

plt.plot(ts)
rolmean = ts.rolling(12).mean()

rolstd = ts.rolling(12).std()



plt.figure(figsize=(20,10))

plt.plot(ts)

plt.plot(rolmean)

plt.plot(rolstd)
ts = df['Users'].dropna()

from statsmodels.tsa.stattools import adfuller



dftest = adfuller(ts, autolag='AIC')

dftest
#let's log the series

plt.figure(figsize=(20,10))

ts_log = np.log(ts)

plt.plot(ts_log)
#Differencing 



ts_log_diff = ts_log - ts_log.shift()



plt.figure(figsize=(20,10))

plt.plot(ts_log_diff)

plt.plot(ts_log)
ts_log_diff = ts_log_diff.dropna()

dftest2 = adfuller(ts_log_diff, autolag='AIC')

dftest2 
plt.figure(figsize=(20,10))

plt.plot(ts_log_diff)
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import acf, pacf



acf = acf(ts_log_diff, nlags = 30)

plt.plot(acf)



plt.axhline(y = 0, linestyle = '-', color = 'orange')

plt.axhline(y = -1.95/np.sqrt(len(ts_log_diff)), linestyle = '-', color = 'red')

plt.axhline(y = 1.95/np.sqrt(len(ts_log_diff)), linestyle = '-', color = 'pink')
pacf = pacf(ts_log_diff, nlags = 30, method = 'ols')

plt.plot(pacf)



plt.axhline(y = 0, linestyle = '-', color = 'yellow')

plt.axhline(y = -1.95/np.sqrt(len(ts_log_diff)), linestyle = '-', color = 'black')

plt.axhline(y = 1.95/np.sqrt(len(ts_log_diff)), linestyle = '-', color = 'green')
#AR 

model = ARIMA(ts_log, order = (2,1,0))

results = model.fit(disp = -1) 



plt.figure(figsize=(20,10))

plt.plot(ts_log_diff)

plt.plot(results.fittedvalues, color = 'red')



RSS = sum((results.fittedvalues - ts_log_diff)**2)

print(RSS)
#MA 

model = ARIMA(ts_log, order = (0,1,2))

results = model.fit(disp = -1) 



plt.figure(figsize=(20,10))

plt.plot(ts_log_diff)

plt.plot(results.fittedvalues, color = 'red')



RSS2 = sum((results.fittedvalues - ts_log_diff)**2)

print(RSS2)
#ARIMA

model = ARIMA(ts_log, order = (2,1,2))

results = model.fit(disp = -1) 



plt.figure(figsize=(20,10))

plt.plot(ts_log_diff)

plt.plot(results.fittedvalues, color = 'red')



RSS3 = sum((results.fittedvalues - ts_log_diff)**2)

print(RSS3)
predictions = pd.Series(results.fittedvalues, copy = True)

predictions_cumsum = predictions.cumsum()

predictions_cumsum
predictions_log = pd.Series(ts_log.iloc[0], index = ts_log.index)

predictions_log = predictions_log.add(predictions_cumsum,fill_value = 0)



predictions_ARIMA = np.exp(predictions_log) 

plt.plot(ts)

plt.plot(predictions_ARIMA)