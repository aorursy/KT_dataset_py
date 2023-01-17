import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv()
dataset.head()
from datetime import datetime
dataset['Month']=pd.to_datetime(dataset['Month'],infer_datetime_format=True)
indexedDataset=dataset.set_index(['Month'])
indexedDataset.head()
plt.xlabel('Date')
plt.ylabel('Total_Usage')
plt.plot(indexedDataset)
rolmean=indexedDataset.rolling(window=12).mean()
rolstd=indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)
orig=plt.plot(indexedDataset,color='blue',label='Original')
mean= plt.plot(rolmean,color='red',label='mean')
std = plt.plot(rolstd,color='black',label='std')
plt.legend(loc='best')
plt.title('Rolling Mean vs Rolling std')
plt.show(block=False)
print('Result of Dickey-Fuller Test: ')
dftest=adfuller(indexedDataset['Total Gallons'],autolag='AIC')
dfoutput= pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)
indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingSTD=indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')
datasetminusMovingAvg=indexedDataset_logScale-movingAverage
datasetminusMovingAvg.head()

#Remove the NaN value
datasetminusMovingAvg.dropna(inplace=True)
datasetminusMovingAvg.head(10)
def test_stationary(timeseries):
    
    #Determine rolling Statistics
    movingAverage=timeseries.rolling(window=12).mean()
    movingStd=timeseries.rolling(window=12).std()
    
    #plotting the rolling statistics
    orig=plt.plot(indexedDataset,color='blue',label='Original')
    mean= plt.plot(rolmean,color='red',label='mean')
    std = plt.plot(rolstd,color='black',label='std')
    plt.legend(loc='best')
    plt.title('Rolling Mean vs Rolling std')
    plt.show(block=False)
    
    #performing the Dickey-Fuller Test
    dftest=adfuller(indexedDataset['Total Gallons'],autolag='AIC')
    dfoutput= pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationary(datasetminusMovingAvg)
exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red')
decomposition=seasonal_decompose(indexedDataset_logScale)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)
#ACF and PACF plots:

lag_acf = acf(datasetLogDiffShifting, nlags=10)
lag_pacf = pacf(datasetLogDiffShifting, nlags=10, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='black')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='black')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
model =ARIMA(indexedDataset_logScale, order=(2,1,2))
results_ARIMA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting['Total Gallons'])**2))
print('Plotting the ARIMA Model')
prediction_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)
prediction_ARIMA_diff.head()
prediction_ARIMA_diff_cumsum=prediction_ARIMA_diff.cumsum()
prediction_ARIMA_diff_cumsum.head()
prediction_ARIMA_log=pd.Series(indexedDataset_logScale['Total Gallons'].ix[0], index=indexedDataset_logScale.index)
prediction_ARIMA_log=prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()
prediction_ARIMA=np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)