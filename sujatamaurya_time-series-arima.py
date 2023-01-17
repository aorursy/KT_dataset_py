from datetime import datetime

import numpy as np             #for numerical computations like log,exp,sqrt etc

import pandas as pd            #for reading & storing data, pre-processing

import matplotlib.pylab as plt #for visualization

#for making sure matplotlib plots are generated in Jupyter notebook itself

%matplotlib inline             

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6
path = "../input/energy/EnergyDemand.csv" #For Kaggle

dataset = pd.read_csv(path)

#Parse strings to datetime type

dataset['Month'] = pd.to_datetime(dataset['Month'],infer_datetime_format=True) #convert from string to datetime

indexedDataset = dataset.set_index(['Month'])

indexedDataset.head(5)
## plot graph

plt.xlabel('Date')

plt.ylabel('Energy Demand')

plt.plot(indexedDataset)
#Determine rolling statistics

rolmean = indexedDataset.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level

rolstd = indexedDataset.rolling(window=12).std()

print(rolmean,rolstd)
#Plot rolling statistics

orig = plt.plot(indexedDataset, color='blue', label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label='Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
#Perform Augmented Dickey–Fuller test:

print('Results of Dickey Fuller Test:')

dftest = adfuller(indexedDataset['#EnergyDemand'], autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
#Estimating trend

indexedDataset_logScale = np.log(indexedDataset)

plt.plot(indexedDataset_logScale)
#The below transformation is required to make series stationary

movingAverage = indexedDataset_logScale.rolling(window=12).mean()

movingSTD = indexedDataset_logScale.rolling(window=12).std()

plt.plot(indexedDataset_logScale)

plt.plot(movingAverage, color='red')
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage

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

    dftest = adfuller(timeseries['#EnergyDemand'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    
test_stationarity(datasetLogScaleMinusMovingAverage)
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.plot(indexedDataset_logScale)

plt.plot(exponentialDecayWeightedAverage, color='red')
datasetLogScaleMinusExponentialMovingAverage = indexedDataset_logScale - exponentialDecayWeightedAverage

test_stationarity(datasetLogScaleMinusExponentialMovingAverage)
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()

plt.plot(datasetLogDiffShifting)
datasetLogDiffShifting.dropna(inplace=True)

test_stationarity(datasetLogDiffShifting)
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
#AR Model

#making order=(2,1,0) gives RSS=1.5023

model = ARIMA(indexedDataset_logScale, order=(2,1,0))

results_AR = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting['#EnergyDemand'])**2))

print('Plotting AR model')
#MA Model

model = ARIMA(indexedDataset_logScale, order=(0,1,2))

results_MA = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting['#EnergyDemand'])**2))

print('Plotting MA model')
# AR+I+MA = ARIMA model

model = ARIMA(indexedDataset_logScale, order=(2,1,2))

results_ARIMA = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['#EnergyDemand'])**2))

print('Plotting ARIMA model')
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())
#Convert to cumulative sum

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log = pd.Series(indexedDataset_logScale['#EnergyDemand'].iloc[0], index=indexedDataset_logScale.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA_log.head()
# Inverse of log is exp.

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(indexedDataset)

plt.plot(predictions_ARIMA)
indexedDataset_logScale
#We have 144(existing data of 12 yrs in months) data points. 

#And we want to forecast for additional 120 data points or 10 yrs.

results_ARIMA.plot_predict(1,264) 

#x=results_ARIMA.forecast(steps=120)
#print(x[1])

#print(len(x[1]))

#print(np.exp(x[1]))