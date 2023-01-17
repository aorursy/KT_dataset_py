import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf 

from statsmodels.tsa.arima_model import ARIMA



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv')

df.head()
df.drop(['realtime_start','realtime_end'],axis=1,inplace=True)

df.head()
df.dtypes
df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')

df.set_index('date',inplace=True)

df.head()
plt.rcParams['figure.figsize'] = (12,6)

plt.plot(df['value'])
rolmean = df.rolling(window=12).mean()

rolstd = df.rolling(window=12).std()



plt.plot(rolmean,color='red',label='Rolling avg')

plt.plot(rolstd,color='black',label='Rolling std')

plt.legend(loc='best')

plt.show()
def test_stationarity(data):

  rolmean = data.rolling(window=12).mean()

  rolstd = data.rolling(window=12).std()



  plt.plot(data,label='Original data')

  plt.plot(rolmean,color='red',label='Rolling avg')

  plt.plot(rolstd,color='black',label='Rolling std')

  plt.legend(loc='best')

  plt.show()



  print('Results of the Dickey Fuller test:')

  dftest = adfuller(data, autolag='AIC')

  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])



  for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

  print (dfoutput)
dfMA = df.rolling(window=12).mean()

dfMAdiff = df - dfMA
test_stationarity(dfMAdiff['value'].dropna())
dfMAdiffshift = dfMAdiff - dfMAdiff.shift(1)

test_stationarity(dfMAdiffshift['value'].dropna())
decomposition = seasonal_decompose(dfMAdiffshift['value'].dropna())



trend =decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411) 

plt.plot(dfMAdiffshift, label='Original') 

plt.legend(loc='best') 

plt.subplot(412) 

plt.plot(trend, label='Trend') 

plt.legend(loc='best') 

plt.subplot(413) 

plt.plot(seasonal,label='Seasonality') 

plt.legend(loc='best') 

plt.subplot(414) 

plt.plot(residual, label='Residuals') 

plt.legend(loc='best') 

plt.tight_layout() 

plt.show()
lag_acf = acf(dfMAdiffshift.dropna(), nlags=12) 

lag_pacf = pacf(dfMAdiffshift.dropna(), nlags=12, method='ols')



plt.plot(lag_acf) 

plt.axhline(y=0,linestyle='--',color='gray') 

plt.axhline(y=-1.96/np.sqrt(len(dfMAdiffshift.dropna())),linestyle='--',color='gray') 

plt.axhline(y=1.96/np.sqrt(len(dfMAdiffshift.dropna())),linestyle='--',color='gray') 

plt.title('Autocorrelation Function') 

plt.show() 

plt.plot(lag_pacf) 

plt.axhline(y=0,linestyle='--',color='gray') 

plt.axhline(y=-1.96/np.sqrt(len(dfMAdiffshift.dropna())),linestyle='--',color='gray') 

plt.axhline(y=1.96/np.sqrt(len(dfMAdiffshift.dropna())),linestyle='--',color='gray') 

plt.title('Partial Autocorrelation Function') 

plt.show()
dfMAdiffshift.dropna(inplace=True)



model = ARIMA(dfMAdiff.dropna(),order=(2,1,3))

result = model.fit()



plt.plot(dfMAdiffshift.dropna(),label='Original')

plt.plot(result.fittedvalues,label='fit')

plt.legend(loc='best')
result.summary()
predictions_ARIMA_diff = pd.Series(result.fittedvalues,copy = True)

print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_diff_cumsum.head()
predictions_ARIMA = pd.Series(dfMAdiff['value'].dropna().iloc[0],index=df.index)

predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0)

print(predictions_ARIMA.head())
plt.plot(df,color='blue',label='Original data')

plt.plot(predictions_ARIMA,color='red',label='Fit')

plt.show()
modelsarimax = sm.tsa.statespace.SARIMAX(df['value'],order=(2,1,3),seasonal_order=(2,1,3,12)).fit()

resultsarimax = modelsarimax.predict(start='2020-01-01',end='2031-01-01',freq='MS',dynamic=True)

plt.plot(df['value'],color='blue',label='Original data')

plt.plot(resultsarimax,color='red',label='Predicted data')

plt.legend(loc='best')
forecast = pd.Series(resultsarimax)

forecast