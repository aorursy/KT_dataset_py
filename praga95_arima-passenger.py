# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/AirPassengers.csv',  index_col='Month',  parse_dates = ['Month'])
data.head()
data.dtypes
data.index
data.plot()
ts = data["#Passengers"] 
ts.head(10)
ts[datetime(1949,8,1)]
ts['1949']
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(8,5))
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
test_stationarity(ts)
ts_log = np.log(ts)
plt.plot(ts_log)
ts_smooth = ts_log.rolling(window = 12).mean()
plt.plot(ts_smooth, color = 'red')
plt.plot(ts_log)
plt.show()
no_sub_ts = ts_smooth
no_sub_ts.dropna(inplace = True)
test_stationarity(no_sub_ts)
sub_ts = ts_log - ts_smooth
sub_ts.dropna(inplace = True)
test_stationarity(sub_ts)
expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(expwighted_avg, color='red')
plt.plot(ts_log)
no_sub_ts = expwighted_avg
no_sub_ts.dropna(inplace = True)
test_stationarity(no_sub_ts)
exp_ts_diff = ts_log-expwighted_avg
exp_ts_diff.dropna(inplace = True)
test_stationarity(exp_ts_diff)

ts_diff = ts_log - ts_log.shift()
ts_diff.dropna(inplace = True)
test_stationarity(ts_diff)
ts_diff_exp = ts_diff  - ts_diff.ewm(halflife = 12).mean()
ts_diff_exp.dropna(inplace = True)
test_stationarity(ts_diff_exp)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

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
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts_diff,lags=20,alpha=1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plot_pacf(ts_diff,lags=20,alpha=1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
from statsmodels.tsa.arima_model import ARMA
mod = ARMA(ts_diff, order=(1,0))
res = mod.fit()
print("The AIC for an AR(1) is: ", res.aic)

# Fit the data to an AR(2) model and print AIC:
mod = ARMA(ts_diff, order=(2,0))
res = mod.fit()
print("The AIC for an AR(2) is: ", res.aic)

mod = ARMA(ts_diff, order=(3,0))
res = mod.fit()
print("The AIC for an AR(3) is: ", res.aic)

# Fit the data to an MA(1) model and print AIC:
mod = ARMA(ts_diff, order=(0,1))
res = mod.fit()
print("The AIC for an MA(1) is: ", res.aic)

mod = ARMA(ts_diff, order=(0,2))
res = mod.fit()
print("The AIC for an MA(2) is: ", res.aic)

mod = ARMA(ts_diff, order=(0,3))
res = mod.fit()
print("The AIC for an MA(3) is: ", res.aic)

# Fit the data to an ARMA(1,1) model and print AIC:
mod = ARMA(ts_diff, order=(1,1))
res = mod.fit()
print("The AIC for an ARMA(1,1) is: ", res.aic)

mod = ARMA(ts_diff, order=(2,2))
res = mod.fit()
print("The AIC for an ARMA(2,2) is: ", res.aic)

mod = ARMA(ts_diff, order=(3,3))
res = mod.fit()
print("The AIC for an ARMA(3,3) is: ", res.aic)
model=pd.DataFrame()
names=['AR(1)','AR(2)','AR(3)','MA(1)','MA(2)','MA(3)','ARMA(1,1)','ARMA(2,2)','ARMA(3,3)']
aic=[-235.38589888263135,-237.6046356975284,-236.95178478978522,-237.5073149855421,-240.3789540202477,-257.8902625951644,-241.60771402612232,-287.2808079212986,-289.4747225981437]
model['Model Name']=names
model['AIC']=aic
model=model.set_index('Model Name')
model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_diff)**2))
model = ARIMA(ts_log, order=(0, 1, 2))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_diff)**2))
model = ARIMA(ts_log, order=(2, 1, 2))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_diff)**2))
preds = pd.Series(results_AR.fittedvalues, copy = True)
preds_cumsum = preds.cumsum()
print (preds_cumsum.head())
preds_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
preds_log = preds_log.add(preds_cumsum,fill_value=0)
preds_log.head()
preds_ARIMA = np.exp(preds_log)
plt.plot(ts)
plt.plot(preds_ARIMA)
plt.xlabel('Years')
plt.ylabel("Number of Passengers")
plt.title('RMSE: %.4f'% np.sqrt(sum((preds_ARIMA-ts)**2)/len(ts)))
results_AR.plot_predict(start='1953-07-01', end='1962-12-01')
plt.show()
model = ARIMA(ts_log, order=(1, 1, 1))  
results_AR_3 = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_AR_3.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR_3.fittedvalues-ts_diff)**2))
preds = pd.Series(results_AR_3.fittedvalues, copy = True)
preds_cumsum = preds.cumsum()
print (preds_cumsum.head())
preds_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
preds_log = preds_log.add(preds_cumsum,fill_value=0)
preds_log.head()
preds_ARIMA = np.exp(preds_log)
plt.plot(ts)
plt.plot(preds_ARIMA)
plt.xlabel('Years')
plt.ylabel("Number of Passengers")
plt.title('RMSE: %.4f'% np.sqrt(sum((preds_ARIMA-ts)**2)/len(ts)))
results_AR_3.plot_predict(start='1953-07-01', end='1962-12-01')
plt.show()