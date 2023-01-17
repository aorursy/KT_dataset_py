import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.pylab import rcParams
from datetime import datetime
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
sns.set_style('darkgrid')
rcParams['figure.figsize'] = 10,6
dateparse = lambda x: datetime.strptime(x, '%Y-%m')
df = pd.read_csv('../input/AirPassengers.csv')
df.head()
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace = True)
plt.plot(df['#Passengers'], linewidth = 3)
plt.xlabel('Years')
plt.ylabel('Number of Passengers')
plt.show()
#Now the rolling statistics
rolmean = df.rolling(window= 12).mean() #Gives a series of means of the number of previous values equals the window size.
print(rolmean.head(20))
rolstd = df.rolling(window=12).std()
print(rolstd.head(20))
plt.plot(df['#Passengers'], linewidth = 2, label = 'Original')
plt.plot(rolmean, linewidth = 2, label = 'Rolling Mean', color = 'r')
plt.plot(rolstd, linewidth = 2, label = 'Rolling Std Dev', color = 'k')
plt.legend(loc = 'best')
plt.title('Rolling Mean and Standard Deviation')
plt.show()
#Performing Augumented Dickey Fuller Test
print('Results of the Dickey Fuller Test')
dftest = adfuller(x = df['#Passengers'], autolag= 'AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)
for key,value in dftest[4].items():
    print('Critical Value ({}) = {}'.format(key,value))
df_log = np.log(df)
plt.plot(df_log['#Passengers'], linewidth = 3)
plt.xlabel('Years')
plt.ylabel('Number of Passengers')
plt.show()
#Tranformation to make stationary
movingAverage = df_log.rolling(window= 12).mean()
movingStd = df_log.rolling(window=12).std()
plt.plot(df_log['#Passengers'], linewidth = 2, label = 'Original')
plt.plot(movingAverage, linewidth = 2, label = 'Rolling Mean', color = 'r')
plt.legend(loc = 'best')
plt.title('Moving Average')
plt.show()
dfLogScaleMinusMovingAverage = df_log - movingAverage
#Removing the NaN Values
dfLogScaleMinusMovingAverage.dropna(inplace= True)
def test_stationary(timeseries):
    #Determining Rolling Statistics
    movingAverage = timeseries.rolling(window = 12).mean()
    movingStd = timeseries.rolling(window = 12).std()
    
    #Plotting Rolling Statistics
    plt.plot(timeseries, linewidth = 2, label = 'Original')
    plt.plot(movingAverage, linewidth = 2, label = 'Rolling Mean', color = 'r')
    plt.plot(movingStd, linewidth = 2, label = 'Rolling Std Dev', color = 'k')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
    
    #Performing Dickey Fuller Test
    #Performing Augumented Dickey Fuller Test
    print('Results of the Dickey Fuller Test')
    dftest = adfuller(x = timeseries['#Passengers'], autolag= 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print(dfoutput)
    for key,value in dftest[4].items():
        print('Critical Value ({}) = {}'.format(key,value))
test_stationary(dfLogScaleMinusMovingAverage)
exDecayWeightedAverage = df_log.ewm(halflife= 12, min_periods=0, adjust= True).mean()
plt.plot(df_log, linewidth = 2, label = 'Log Transformation')
plt.plot(exDecayWeightedAverage, linewidth = 2, color = 'r', label = 'Exponential Decay')
plt.title("Exponential Scale and Log Scale")
plt.legend()
plt.show()
dfLogScaleMinusExpoMovingAverage =  df_log - exDecayWeightedAverage
test_stationary(dfLogScaleMinusExpoMovingAverage)
dfLogTimeShift = df_log.shift()
dfLogDiffShift = df_log - dfLogTimeShift 
plt.plot(dfLogDiffShift, linewidth = 2)
plt.show()
dfLogDiffShift.dropna(inplace= True)
test_stationary(dfLogDiffShift)
decomposition = seasonal_decompose(df_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.plot(df_log, label = 'Original', linewidth = 2)
plt.plot(trend, label = 'Trend', linewidth = 2)
plt.plot(residual, label = 'Residual', linewidth = 2)
plt.plot(seasonal,label = 'seasonal', linewidth = 2)
plt.legend()
plt.show()

decomposedLog = residual
decomposedLog.dropna(inplace = True)
test_stationary(decomposedLog)
lag_acf = acf(dfLogDiffShift, nlags = 20)
lag_pacf = pacf(dfLogDiffShift, nlags = 20, method= 'ols')

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf, linewidth = 2)
plt.axhline(y = 0, linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(dfLogDiffShift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dfLogDiffShift)), linestyle='--', color='gray')
plt.title('Autocorrelation Function') 

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dfLogDiffShift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dfLogDiffShift)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout() 
model = ARIMA(df_log, order=(2,1,0))
results_AR = model.fit(disp = -1)
plt.plot(dfLogDiffShift)
plt.plot(results_AR.fittedvalues, color = 'r')
plt.title('RSS: {:.4f}'.format(sum((results_AR.fittedvalues - dfLogDiffShift['#Passengers'])**2)))
print('Plotting AR model')
model = ARIMA(df_log, order=(0,1,2))
results_MA = model.fit(disp = -1)
plt.plot(dfLogDiffShift)
plt.plot(results_AR.fittedvalues, color = 'r')
plt.title('RSS: {:.4f}'.format(sum((results_AR.fittedvalues - dfLogDiffShift['#Passengers'])**2)))
print('Plotting MA model')
model = ARIMA(df_log, order=(2,1,2))
results_ARIMA = model.fit(disp = -1)
plt.plot(dfLogDiffShift)
plt.plot(results_AR.fittedvalues, color = 'r')
plt.title('RSS: {:.4f}'.format(sum((results_AR.fittedvalues - dfLogDiffShift['#Passengers'])**2)))
print('Plotting ARIMA model')
predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy= True)
print(predictions_ARIMA.head())
#Convet to cumulative sum
predictions_ARIMA_cumsum = predictions_ARIMA.cumsum()
predictions_ARIMA_cumsum.head(10)
predictions_ARIMA_log = pd.Series(df_log['#Passengers'].iloc[0], index = df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_cumsum, fill_value = 0)
predictions_ARIMA_log.head()
#Inverse of log is exponent
plt.figure(figsize=(13,8))
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df, linewidth = 2, label = 'Original')
plt.plot(predictions_ARIMA, linewidth = 2, label = 'Prediction')
plt.legend()
plt.show()
df_log
plt.figure(figsize=(13,8))
results_ARIMA.plot_predict(1,264)
plt.legend(loc = 'upper left')
plt.show()