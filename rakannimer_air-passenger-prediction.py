# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6



dataset = pd.read_csv("../input/AirPassengers.csv")

# Parse strings to datetime type

dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)

indexedDataset = dataset.set_index(['Month'])



from datetime import datetime

indexedDataset['1949-03']

indexedDataset['1949-03':'1949-06']

indexedDataset['1949']
plt.xlabel("Date")

plt.ylabel("Number of air passengers")

plt.plot(indexedDataset)
#Determing rolling statistics

rolmean = indexedDataset.rolling(window=12).mean()

rolstd = indexedDataset.rolling(window=12).std()

print(rolmean, rolstd)
#Plot rolling statistics:

orig = plt.plot(indexedDataset, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)

    
#Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller



print ('Results of Dickey-Fuller Test:')

dftest = adfuller(indexedDataset['#Passengers'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
# Estimating trend

indexedDataset_logScale = np.log(indexedDataset)

plt.plot(indexedDataset_logScale)
movingAverage = indexedDataset_logScale.rolling(window=12).mean()

movingSTD = indexedDataset_logScale.rolling(window=12).std()

plt.plot(indexedDataset_logScale)

plt.plot(movingAverage, color='red')
# Get the difference between the moving average and the actual number of passengers

datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage

datasetLogScaleMinusMovingAverage.head(12)

#Remove Nan Values

datasetLogScaleMinusMovingAverage.dropna(inplace=True)

datasetLogScaleMinusMovingAverage.head(10)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    movingAverage = timeseries.rolling(window=12).mean()

    movingSTD = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')

    std = plt.plot(movingSTD, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries['#Passengers'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(datasetLogScaleMinusMovingAverage)
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.plot(indexedDataset_logScale)

plt.plot(exponentialDecayWeightedAverage, color='red')
datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage

test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()

plt.plot(datasetLogDiffShifting)
datasetLogDiffShifting.dropna(inplace=True)

test_stationarity(datasetLogDiffShifting)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(indexedDataset_logScale)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(indexedDataset_logScale, label='Original')

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
decomposedLogData = residual

decomposedLogData.dropna(inplace=True)

test_stationarity(decomposedLogData)
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(datasetLogDiffShifting, nlags=20)

lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')





#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA



#AR MODEL

model = ARIMA(indexedDataset_logScale, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(datasetLogDiffShifting)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))

print('Plotting AR model')
#MA MODEL

model = ARIMA(indexedDataset_logScale, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.plot(datasetLogDiffShifting)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))

print('Plotting AR model')
model = ARIMA(indexedDataset_logScale, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(datasetLogDiffShifting)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print (predictions_ARIMA_diff.head())
#Convert to cumulative sum

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print (predictions_ARIMA_diff_cumsum.head())
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)

#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

#predictions_ARIMA_log.head()



predictions_ARIMA_log = pd.Series(indexedDataset_logScale['#Passengers'].ix[0], index=indexedDataset_logScale['#Passengers'].index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(indexedDataset)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-indexedDataset["#Passengers"])**2)/len(indexedDataset["#Passengers"])))