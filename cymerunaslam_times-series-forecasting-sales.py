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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

raw = pd.read_csv("/kaggle/input/Iowa_Liqor.csv", sep='\t')
raw.drop(["Date.1"], axis=1, inplace=True)
raw.tail()
raw.info()
from datetime import datetime

raw.Date = pd.to_datetime(raw.Date) 
raw.info()
liqor = raw.set_index(raw['Date'])

liqor = liqor.sort_index()
liqor["2015"].tail()
liqor.shape
ls = liqor["Sale"].resample("M").sum()
ls.shape
plt.figure(figsize = (20, 10))

plt.plot(ls, color = "blue", label = "Monthly sales")

plt.title("Monthly sales")

plt.legend();
from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries):

    

    #Determing rolling statistics

#     rolmean = pd.rolling_mean(timeseries, window=12)

#     rolstd = pd.rolling_std(timeseries, window=12)

    

    rolmean = pd.Series(timeseries).rolling(window=12).mean()

    rolstd = pd.Series(timeseries).rolling(window=12).std()



    #Plot rolling statistics:

    plt.figure(figsize = (20, 10))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(ls)
ls_log = np.log(ls)



plt.figure(figsize = (20, 10))

plt.plot(ls_log, color = "blue", label = "Monthly sales")

plt.title("Log Transformation Monthly sales")

plt.legend();

moving_avg = pd.Series(ls_log).rolling(window=12).mean()



plt.figure(figsize = (20, 10))



plt.plot(ls_log)

plt.plot(moving_avg, color='red')

plt.title('Moving Average')

plt.show()
ls_log_moving_avg_diff = ls_log - moving_avg

print(ls_log_moving_avg_diff.head(12))
ls_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(ls_log_moving_avg_diff)


expwighted_avg = ls_log.ewm(halflife=12).mean()



plt.figure(figsize = (20, 10))



plt.plot(ls_log)

plt.plot(expwighted_avg, color='red')

plt.title('Exponentially Weighted Average')

plt.show()

ls_log_ewma_diff = ls_log - expwighted_avg

test_stationarity(ls_log_ewma_diff)
ls_log_diff = ls_log - ls_log.shift()



plt.figure(figsize = (15, 10))



plt.plot(ls_log_diff)
ls_log_diff.dropna(inplace=True)

test_stationarity(ls_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ls_log)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.figure(figsize = (15, 10))

plt.subplot(411)

plt.plot(ls_log, label='Original')

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
ls_log_decompose = residual

ls_log_decompose.dropna(inplace=True)

test_stationarity(ls_log_decompose)
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ls_log_diff, nlags=20)

lag_pacf = pacf(ls_log_diff, nlags=20, method='ols')
#Plot ACF: 



plt.figure(figsize = (15, 10))



plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ls_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ls_log_diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ls_log_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ls_log_diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
#AR Model

model = ARIMA(ls_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.figure(figsize = (15, 10))

plt.plot(ls_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ls_log_diff)**2))
#MA model

model = ARIMA(ls_log, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.figure(figsize = (15, 10))

plt.plot(ls_log_diff)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ls_log_diff)**2))
model = ARIMA(ls_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)

plt.figure(figsize = (15, 10))

plt.plot(ls_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ls_log_diff)**2))
predictions_AR_diff = pd.Series(results_AR.fittedvalues, copy=True)

predictions_AR_diff.head()
predictions_AR_diff_cumsum = predictions_AR_diff.cumsum()

predictions_AR_diff_cumsum.head()
#predictions_ARIMA_log = pd.Series(ls_log.loc[0], index=ls_log.index)

predictions_AR_log = pd.Series(ls_log.loc[:], index=ls_log.index)

predictions_AR_log = predictions_AR_log.add(predictions_AR_diff_cumsum,fill_value=0)

predictions_AR_log.head()
# predictions_ARIMA_log = pd.Series(indexedDataset_logScale['#Passengers'].iloc[0], index=indexedDataset_logScale.index)

# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

# predictions_ARIMA_log.head()
predictions_AR = np.exp(predictions_AR_log)

plt.figure(figsize = (15, 10))

plt.plot(ls)

plt.plot(predictions_AR)

plt.title('RMSE: %.2f'% np.sqrt(sum((predictions_AR-ls)**2)/len(ls)))
plt.figure(figsize = (20, 10))

results_AR.plot_predict(1,200) 