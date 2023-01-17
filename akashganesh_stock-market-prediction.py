import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

djia= pd.read_csv('../input/stocknews/upload_DJIA_table.csv',parse_dates=['Date'], index_col='Date')
djia.head()

ts=djia['Open']
plt.plot(ts)
plt.show()
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    # Determining Rolling Statistics
    rolmean=timeseries.rolling(window=12).mean()
    rolstd=timeseries.rolling(window=12).std()
    
    #Plot Rolling Statistics
    orig=plt.plot(timeseries,color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Plot Dickey-Fuller Test
    
    print('Results of Dickey-Fuller Test:')
    dftest=adfuller(timeseries , autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value
    print (dfoutput)
    
test_stationarity(ts)
#taking log transformation

ts_log=np.log(ts)

plt.plot(ts_log)
import matplotlib.gridspec as gridspec

ts_log = np.log(ts)
fig = plt.figure(constrained_layout = True)
gs_1 = gridspec.GridSpec(2, 3, figure = fig)
ax_1 = fig.add_subplot(gs_1[0, :])
ax_1.plot(ts_log)
ax_1.set_xlabel('Year')
ax_1.set_ylabel('Data')
plt.title('Logarithmic time series')

ax_2 = fig.add_subplot(gs_1[1, :])
ax_2.plot(ts)
ax_1.set_xlabel('Year')
ax_1.set_ylabel('Data')
plt.title('Original time series')
plt.show()
mov_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(mov_avg, color='red')
from sklearn import datasets, linear_model

ts_wi = ts_log.reset_index()
df_values = ts_wi.values
train_y = df_values[:,1]
train_y = train_y[:, np.newaxis]
train_x = ts_wi.index
train_x = train_x[:, np.newaxis]
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
pred = regr.predict(train_x)
plt.plot(ts_wi.Date, pred)
plt.plot(ts_log)
ts_log_moving_avg_diff= ts_log-mov_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=4, model='additive')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
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
#ts_decompose = residual
ts_log_diff = ts_log - ts_log.shift(1)
ts_decompose = ts_log_diff
ts_decompose.dropna(inplace=True)
test_stationarity(ts_decompose)
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))