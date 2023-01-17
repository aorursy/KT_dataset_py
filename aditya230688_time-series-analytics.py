import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6

dataset = pd.read_csv('../input/airline.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset= dataset.set_index(['Month'])

from datetime import datetime
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    ##Determining Rolling Statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    ##Plot Rolling Statistics
    orig = plt.plot(timeseries, color = 'blue',label = 'Original')
    mean = plt.plot(movingAverage, color = 'red',label = 'Rolling Mean')
    std = plt.plot(movingSTD, color = 'black',label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    ##Perform Dickey Fuller Test
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['Airline Passenger'], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(indexedDataset_logScale)
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
#plt.plot(datasetLogDiffShifting)
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()
# Now we need to calculate the value of P and Q from the PACF and ACF graph respecively
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method = 'ols')

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')
plt.title('Auto Correlation Function')
plt.tight_layout()

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='grey')
plt.title('Partial Auto Correlation Function')
plt.tight_layout()

##From the below graph, value of P = 2 (from PACF) and value of Q = 2 (from ACF). 
##These values are the intersection points of value at 0
from statsmodels.tsa.arima_model import ARIMA

#ARIMA Model
model = ARIMA(indexedDataset_logScale, order=(2,1,2))
# order=(2,1,2) - order=(P,d,Q)
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' %sum((results_ARIMA.fittedvalues - datasetLogDiffShifting["Airline Passenger"])**2))
## RSS - Residual Sum of Squares
print('Plotting ARIMA Model')
prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)
#print(prediction_ARIMA_diff.head())
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
#print(prediction_ARIMA_diff_cumsum.head())
prediction_ARIMA_log = pd.Series(indexedDataset_logScale['Airline Passenger'].ix[0], index=indexedDataset_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)
results_ARIMA.plot_predict(1,264)
results_ARIMA.forecast(steps=120)
prediction_ARIMA.to_csv('pred_val.csv', encoding='utf-8',header=['Airline Passenger'])
dataset = pd.read_csv("pred_val.csv")
#Parse strings to date time type
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
pred_val_dataset= dataset.set_index(['Month'])
Diff_val=pred_val_dataset.subtract(indexedDataset, fill_value=0)
Diff_val[['Airline Passenger']]=Diff_val[['Airline Passenger']].astype(int)
Diff_val.head(20)
