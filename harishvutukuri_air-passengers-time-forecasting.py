import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10
import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv("../input/AirPassengers.csv")
dataset.head()
# Convert string to DateTime format
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])
from datetime import datetime
indexedDataset.head()
# plot graph

plt.xlabel("Date")
plt.ylabel("No of Passengers")
plt.plot(indexedDataset)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
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

test_stationarity(indexedDataset)
## Though the variation in standard deviation is small, mean is clearly increasing with time and this is not a stationary series.
## Also, the test statistic is way more than the critical values.
# Estimated trend
indexedDataset_log = np.log(indexedDataset)
plt.plot(indexedDataset_log)
# calculating moving average for the log scale dataset
movingAverage = indexedDataset_log.rolling(window = 12).mean()
movingSTD = indexedDataset_log.rolling(window = 12).std()

plt.plot(indexedDataset_log)
plt.plot(movingAverage, color='red')
# minus moving average
indexedDataset_log_Minus_MovingAverage = indexedDataset_log - movingAverage
indexedDataset_log_Minus_MovingAverage.head(12)
# remove nan
indexedDataset_log_Minus_MovingAverage.dropna(inplace = True)
indexedDataset_log_Minus_MovingAverage.head(12)
# again test for stationarity
test_stationarity(indexedDataset_log_Minus_MovingAverage)

## Test Statistic and Critical Value are almost equal ==> stationary
## exponentially weighted moving average where weights are assigned to all the previous values with a decay factor

exponentialDecayWeightedAverage = indexedDataset_log.ewm(halflife = 12, min_periods = 0, adjust = True).mean()
plt.plot(indexedDataset_log)
plt.plot(exponentialDecayWeightedAverage, color='red')
# minus the exponential decay average
indexedDataset_log_Minus_ExponentialDecayAverage = indexedDataset_log - exponentialDecayWeightedAverage
test_stationarity(indexedDataset_log_Minus_ExponentialDecayAverage)
## This TS has even lesser variations in mean and standard deviation in magnitude. 
##Also, the test statistic is smaller than the 1% critical value, which is better than the previous case
# Shifting the values to Time Series (shifted by value 1, so for ARIMA d = 1)
datasetLogDiffShifting = indexedDataset_log - indexedDataset_log.shift()
plt.plot(datasetLogDiffShifting)
# dropping na values and testing for stationary
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
# We can see that the mean and std variations have small variations with time. 
# Also, the Dickey-Fuller test statistic is less than the 10% critical value, thus the TS is stationary with 90% confidence
## Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_log, label='Original')
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
decomposedLog = residual
decomposedLog.dropna(inplace= True)
test_stationarity(decomposedLog)
# Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) graphs for Q and P values in ARIMA

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
## The lag value where the chart crosses the upper confidence interval for the first time. 
## By this we get P = 2 and Q = 2
# ARIMA Model (p,d,q)

from statsmodels.tsa.arima_model import ARIMA

# AR Model

model = ARIMA(indexedDataset_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('AR Model - RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting['#Passengers'])**2))
#MA Model

model = ARIMA(indexedDataset_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('MA Model - RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting['#Passengers'])**2))
# Combined Model

model = ARIMA(indexedDataset_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('Combined Model - RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting['#Passengers'])**2))
## Here we can see that the AR and MA models have almost the same RSS but combined is significantly better. 
# Converting to Original Scale

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
# remove the lag '1' we have added 

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()
#  add them to base number

predictions_ARIMA_log = pd.Series(indexedDataset_log['#Passengers'].iloc[0], index=indexedDataset_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
# Last step is to take the exponent and compare with the original series
from sklearn.metrics import mean_squared_error, r2_score

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.title('RMSE: {0}.4f'.format(np.sqrt(mean_squared_error(indexedDataset, predictions_ARIMA))))
print(r2_score(indexedDataset, predictions_ARIMA))
## We got RMSE value of 90
indexedDataset.shape
# To forecast next 10 years = 120 months. Then add 120 to 144 = 264

# plot the predictions
results_ARIMA.plot_predict(1,264)

# prediction values for 10 year
prediction_values = results_ARIMA.forecast(steps=120)