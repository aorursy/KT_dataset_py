import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6
dataset = pd.read_excel(r'../input/sku-forecasting-dataset/data.xlsx')
dataset.head()
dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format=True)
indexeddataset = dataset.set_index(['date'])
indexeddataset.head()
df1_Cucumber = indexeddataset[indexeddataset['sku']=='Cucumber (Indian)']
df1_Cucumber = df1_Cucumber[['sales']]

# Plot graph

plt.figure(figsize=(8,6))
plt.xlabel('Date')
plt.ylabel('Sales')
plt.plot(df1_Cucumber)
plt.show()
rolmean = df1_Cucumber.rolling(window=12).mean()
rolstd = df1_Cucumber.rolling(window=12).std()
print(rolmean,rolstd)
orig = plt.plot(df1_Cucumber,color='blue',label='Original')
mean = plt.plot(rolmean,color='red',label='Rolling Mean')
std  = plt.plot(df1_Cucumber,color='green',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Std Dev')
plt.show(block=False)
plt.show()
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey Fuller Test:')
dftest = adfuller(df1_Cucumber['sales'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','P-Value','#Lags Used','No. of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
df1_Cucumber_logscale = np.log(df1_Cucumber)
plt.plot(df1_Cucumber_logscale)
plt.show()
movingAverage = df1_Cucumber_logscale.rolling(window=12).mean()
movingSTD = df1_Cucumber_logscale.rolling(window=12).std()
plt.plot(df1_Cucumber_logscale)
plt.plot(movingAverage, color='red')
plt.show()
df1_Cucumber_logscale_MinusMovingAverage = df1_Cucumber_logscale - movingAverage
df1_Cucumber_logscale_MinusMovingAverage.head()

# Remove NaN values:

df1_Cucumber_logscale_MinusMovingAverage.dropna(inplace=True)
df1_Cucumber_logscale_MinusMovingAverage.head()

from statsmodels.tsa.stattools import adfuller 
def test_stationarity(timeseries):
    
    # Determing rotting statistics
    
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    # Ptot rotting statistics:
    
    orig = plt.plot(timeseries, color = 'blue',label = 'Original')
    mean = plt.plot(movingAverage, color = 'red', label = 'Rolling Mean') 
    std = plt.plot(movingSTD, color = 'black', label = 'Rolling Std') 
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block = False)
    
    # Perform Dickey-Fuller test:
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['sales'], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df1_Cucumber_logscale_MinusMovingAverage)

weightedAverage = df1_Cucumber_logscale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(df1_Cucumber_logscale)
plt.plot(weightedAverage, color='red')
plt.show()


dataset_logscale_weightedAverage = df1_Cucumber_logscale - weightedAverage
test_stationarity(dataset_logscale_weightedAverage)
datasetLogDiffShift = df1_Cucumber_logscale - df1_Cucumber_logscale.shift()
plt.plot(datasetLogDiffShift)
plt.xticks(rotation=90)
plt.show()
datasetLogDiffShift.dropna(inplace=True)
test_stationarity(datasetLogDiffShift)
from statsmodels.tsa.seasonal import seasonal_decompose 

decomposition = seasonal_decompose(df1_Cucumber_logscale)

trend = decomposition.trend
seasonal = decomposition.seasonal 
residual = decomposition.resid
plt.subplot(411)
plt.plot(df1_Cucumber_logscale, label='Original') 
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



residual.head()
decomposedLogdata = residual
decomposedLogdata.dropna(inplace=True)
datasetLogDiffShift.head()


from statsmodels.tsa.stattools import acf, pacf 

lag_acf = acf(datasetLogDiffShift, nlags=20) 
lag_pacf = pacf(datasetLogDiffShift, nlags=20, method='ols') 

# Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShift)),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShift)),linestyle='--',color='gray') 
plt.title('Autocorrelation Function') 
            
# Plot PACF:
            
plt.subplot(122) 
plt.plot(lag_pacf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShift)),linestyle='--' ,color='gray') 
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShift)),linestyle='--' ,color='gray') 
plt.title('Partial Autocorrelation Function') 
plt.tight_layout() 

datasetLogDiffShift.head()

from statsmodels.tsa.arima_model import ARIMA 

# AR MODEL 

model = ARIMA(df1_Cucumber_logscale, order=(0, 1, 2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShift)
plt.plot(results_AR.fittedvalues, color='red') 
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShift["sales"])**2)) 

print('Plotting AR model') 

model = ARIMA(df1_Cucumber_logscale, order=(0, 1, 2))
results_MA =  model.fit(disp=-1)
plt.plot(datasetLogDiffShift)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShift['sales'])**2))
print('Plotting AR model') 

model = ARIMA(df1_Cucumber_logscale, order=(0, 1, 2))
results_ARIMA =  model.fit(disp=-1)
plt.plot(datasetLogDiffShift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShift['sales'])**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
df1_Cucumber_logscale.head()
predictions_ARIMA_log = pd.Series(df1_Cucumber_logscale['sales'], index=df1_Cucumber_logscale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df1_Cucumber)
plt.plot(predictions_ARIMA)
print(df1_Cucumber_logscale)
print(df1_Cucumber_logscale.shape)
results_ARIMA.plot_predict(1,49)
x  = results_ARIMA.forecast(steps=10)
x[1]
import math
for i in enumerate(x[0]):
    print('Prediction Day-'+ str(i[0]+1)+' ->',end= ' ')
    print(math.exp(i[1]))