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
import pandas as pd

import keras

import sklearn



from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse



data = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv', parse_dates=['date'], index_col='date')

data.reset_index(inplace=True)



data.head(3)
data.index = pd.date_range(freq='D',start=data['date'][0], periods=len(data['date']))

from matplotlib.pyplot import plot



data['value'].plot(color='k', title='Original Series')
result_mul = seasonal_decompose(data['value'], model='multiplicative', extrapolate_trend='freq')

result_mul.plot()
result_add = seasonal_decompose(data['value'], model='additive', extrapolate_trend='freq')

result_add.plot()
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



plot_acf(data['value'].tolist(), lags=50)

#plot_pacf(data['value'].tolist(), lags=50)
from statsmodels.nonparametric.smoothers_lowess import lowess



# 1. Moving Average

df_ma = data['value'].rolling(1000, center=True, closed='both').mean()

df_ma.plot(title='Moving Average')

df_std = data['value'].rolling(12, center=True, closed='both').std()

df_std.plot(title='Moving Average')

data['value'].plot(title='Moving Average')
# 2. Loess Smoothing (5% and 15%)

df_loess_5 = pd.DataFrame(lowess(data['value'], np.arange(len(data['value'])), frac=0.05)[:, 1], index=data['date'], columns=['value'])

df_loess_15 = pd.DataFrame(lowess(data['value'], np.arange(len(data['value'])), frac=0.85)[:, 1], index=data['date'], columns=['value'])



# Plot

df_loess_5['value'].plot(title='Loess Smoothed 5%')

df_loess_15['value'].plot(title='Loess Smoothed 85%')

#Stationarity test

from statsmodels.tsa.stattools import adfuller



print('Results of Dickey-Fuller Test:')

dftest = adfuller(data['value'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
diff = data['value'] - data['value'].shift()

df_ma = diff.rolling(40, center=True, closed='both').mean()

df_ma.plot(title='Moving Average')

df_std = diff.rolling(12, center=True, closed='both').std()

df_std.plot(title='Moving Average')

diff.plot(title='Moving Average')
#seasonality

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pylab as plt



diff.dropna(inplace=True)

decomposition = seasonal_decompose(diff)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.plot(diff,label='Original')

plt.legend(loc='best')

plt.plot(trend,label='Trend')

plt.legend(loc='best')

plt.plot(seasonal,label='Seasonal')

plt.legend(loc='best')

plt.plot(residual,label='Residual')

plt.legend(loc='best')
#let's try to predict now

from statsmodels.tsa.arima_model import ARIMA



diff_trend = diff-trend

diff_trend.dropna(inplace=True)



model = ARIMA(diff_trend,order=(1,1,0))

results_AR = model.fit(disp=-1)



plt.plot(diff_trend,label='Original')

plt.legend(loc='best')

plt.plot(results_AR.fittedvalues,label='fit')

plt.legend(loc='best')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-diff_trend)**2))
model = ARIMA(diff_trend,order=(0,1,2))

results_MA = model.fit(disp=0)



plt.plot(diff_trend,label='Original')

plt.legend(loc='best')

plt.plot(results_MA.fittedvalues,label='fit')

plt.legend(loc='best')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-diff_trend)))
#we can plot the predictions and extrapolate to the future

predictions_ARIMA_diff = pd.Series(results_MA.fittedvalues,copy=True)

plt.plot(predictions_ARIMA_diff,label='Fit')



#how to extrapolate to the future?

date_rng = pd.date_range(start='1952-01-01', end='1958-01-01', freq='D')



output = results_MA.forecast()

print(output)





#create a time series from a random generation

#df = pd.DataFrame(date_rng, columns=['date'])

#df['data'] = np.random.randint(0,100,size=(len(date_rng)))