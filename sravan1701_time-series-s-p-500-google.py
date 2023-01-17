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
train_df=pd.read_csv('../input/all_stocks_5yr.csv')
train_df.head()
train_df.dtypes
#convert the date object to datetime 
train_df['date']=pd.to_datetime(train_df['date'])
train_df.dtypes
import matplotlib.pyplot as plt
import seaborn as sns
google = train_df.loc[train_df['Name'] == 'GOOGL']
google.head()
google = google.drop(google.loc[google['volume'].isnull()].index) #drops rows with a null cell in the Volume column
google = google.drop(google.loc[google['open'].isnull()].index) #drops rows with a null cell in the Open column
#checking the missing values in the data
google.isnull().sum()
#creating the subplots for each of the type
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1 = sns.lineplot(x="date", y="open",
                  markers=True, dashes=False, data=google)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax2 = sns.lineplot(x="date", y="high",
                  markers=True, dashes=False, data=google)
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
ax3 = sns.lineplot(x="date", y="low",
                  markers=True, dashes=False, data=google)
ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
ax4 = sns.lineplot(x="date", y="close",
                  markers=True, dashes=False, data=google)
google_cols=google[['date','open','high','low','close']]
google_cols=google_cols.set_index('date')
#we can see an upward trend for all the data
#lets determine the rolling statastics for the data
#Determine rolling statistics
rolmean = google_cols.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
rolstd = google_cols.rolling(window=12).std()
print(rolmean,rolstd)
#Plot rolling statistics
orig = plt.plot(google_cols['high'], color='blue', label='Original')
mean = plt.plot(rolmean['high'], color='red', label='Rolling Mean')
std = plt.plot(rolstd['high'], color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
#Perform Augmented Dickey–Fuller test:
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
print('Results of Dickey Fuller Test:')
dftest = adfuller(google_cols['high'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)
#to achieve the stationary in the data we should do log transfomation for the above function
#The below transformation is required to make series stationary
google_cols_high=google_cols[['high']]
google_cols_high
google_cols_high_logScale = np.log(google_cols_high)
plt.plot(google_cols_high_logScale)

movingAverage = google_cols_high_logScale.rolling(window=12).mean()
movingSTD = google_cols_high_logScale.rolling(window=12).std()
plt.plot(google_cols_high_logScale)
plt.plot(movingAverage, color='red')
datasetLogScaleMinusMovingAverage =google_cols_high_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)
def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['high'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(datasetLogScaleMinusMovingAverage)
# we can see that there is a increase in p value from 0.91 to 0.00003 but we can see that test stats are not close to the critical values
exponentialDecayWeightedAverage = google_cols_high_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(google_cols_high_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
datasetLogScaleMinusExponentialMovingAverage = google_cols_high_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)
decomposition = seasonal_decompose(google_cols_high_logScale,freq=52) 

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(google_cols_high_logScale, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()

#there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be 
#any residual component & that would be a null or NaN. Hence, we also remove such cases.
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)
#ACF & PACF plots

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')  

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout() 
model = ARIMA(google_cols_high_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['high'])**2))
print('Plotting ARIMA model')
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log = pd.Series(google_cols_high_logScale['high'].iloc[0], index=google_cols_high_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()
# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(google_cols_high_logScale)
plt.plot(predictions_ARIMA)
google_cols_high_logScale
results_ARIMA.plot_predict(1,264) 
x=results_ARIMA.forecast(steps=120)
