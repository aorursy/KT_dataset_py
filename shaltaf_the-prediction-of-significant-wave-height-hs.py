from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

from statsmodels.tsa.arima_model import ARIMA

from pandas import DataFrame

from sklearn.metrics import mean_squared_error

from math import sqrt
print(os.listdir('../input'))
# Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv has 43728 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv',index_col=0, parse_dates=True,nrows=24000)

df1.dataframeName = 'Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

df1.head()
df1=df1.replace(-99.9,np.nan)

df1=df1.interpolate(limit_direction='both')
def plot_df(x, y, title, xlabel, ylabel, dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()



# Create Training and Test

train = df1.iloc[:3000,1]

test = df1.iloc[3000:3200,1]
plot_df( train.index, train.values, "significant wave height (SWH or Hs)", "date-time", "Hs(m)")
result=adfuller(train)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

result_diff = adfuller(train.diff().dropna())

print('ADF Statistic: %f' % result_diff[0])

print('p-value: %f' % result_diff[1])
x=plot_pacf((train.diff()).dropna(),lags=10)
x=plot_acf((train.diff()).dropna(),lags=10)
# Build Model

predictions=[]

history=[x for x in train]

for t in range(len(test)):

    model = ARIMA(history[t:], order=(5, 1, 2))

    model_fit = model.fit()

    predictions.append(model_fit.forecast(alpha=0.05)[0][0])

    history.append(test[t])

# Forecast

# Make as pandas series

predictions = pd.Series(predictions, index=test.index)

# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(predictions, label='forecast')

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()

plt.figure(figsize=(12,5), dpi=100)

plt.plot(test, label='actual')

plt.plot(predictions, label='forecast')

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
mse = mean_squared_error(test, predictions)

rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)

plt.scatter(predictions,test)
x=np.array(test)

y=np.array(predictions)

from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(predictions,test.iloc[:])

print(r_value)