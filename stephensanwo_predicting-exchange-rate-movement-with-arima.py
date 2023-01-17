# Predicting Exchange Rate Movement - ARIMA
import matplotlib.pylab as plt

%matplotlib inline

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

from datetime import datetime, timedelta
from urllib.request import Request, urlopen



site= "https://www.cbn.gov.ng/Functions/export.asp?tablename=exchange"

hdr = {'User-Agent': 'Mozilla/5.0'}

req = Request(site,headers=hdr)

page = urlopen(req)
cbn_rates = pd.read_csv(page, index_col=False)
cbn_rates['Rate Date']=pd.to_datetime(cbn_rates['Rate Date'])
#USD Cleaning



cbn_usd = cbn_rates[cbn_rates.Currency=='US DOLLAR']

cbn_usd.reset_index(drop=True, inplace=True)
#Try except to adjust for data not available on weekends



try:

    cbn_usd_filtered = cbn_usd.iloc[cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(5), '%Y-%m-%d')].index.values[0]:cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(5), '%Y-%m-%d')].index.values[0]+365]

except IndexError:

    cbn_usd_filtered = cbn_usd.iloc[cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')].index.values[0]:cbn_usd[cbn_usd['Rate Date']==datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')].index.values[0]+365]
cbn_usd_filtered.set_index('Rate Date', inplace = True)
cbn_usd_no_outliers = cbn_usd_filtered.loc[cbn_usd_filtered['Central Rate'] < 500]
cbn_usd_no_outliers
#resample to weekly mean

cbn_usd_week = cbn_usd_no_outliers[['Buying Rate', 'Central Rate', 'Selling Rate']].resample('W', how = 'mean', fill_method = 'ffill')
cbn_usd_week.head()
plt.plot(cbn_usd_week.index.to_pydatetime(), cbn_usd_week['Central Rate'].values)
#Function to check for stationarity -  Stationarity of a time series means that the mean and variance doesnt change over time or the joint probability remains constant (i.e. the probability that the points will not change over time and will remain the same over time)

#F(yt) = F(yt+k) - (F is the joint probability) thus if you increase the series by k period, the joint probability of the series should not be affected. learn more about series stationarity via this link https://www.youtube.com/watch?v=xbzJp78kOiI 

# A stationary time series will not have a trend. 





def check_stationarity(timeseries):

    

    #Determing rolling statistics

    rolling_mean = timeseries.rolling(window=52,center=False).mean() 

    rolling_std = timeseries.rolling(window=52,center=False).std()



    #Plot rolling statistics:

    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')

    mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')

    std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dickey_fuller_test = adfuller(timeseries, autolag='AIC')

    dfresults = pd.Series(dickey_fuller_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dickey_fuller_test[4].items():

        dfresults['Critical Value (%s)'%key] = value

    print (dfresults)
check_stationarity(cbn_usd_week['Central Rate'])
#applying nonlinear log transformation

cbn_usd_week_log = np.log(cbn_usd_week)
check_stationarity(cbn_usd_week_log['Central Rate'])
#Removing trend and seasonality with decomposition



decomposition = seasonal_decompose(cbn_usd_week['Central Rate'])



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



# Select the most recent weeks 

cbn_usd_week_log_select = cbn_usd_week_log['Central Rate'][-100:]



plt.subplot(411)

plt.plot(cbn_usd_week_log_select.index.to_pydatetime(), cbn_usd_week_log_select.values, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(cbn_usd_week_log_select.index.to_pydatetime(), trend[-100:].values, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(cbn_usd_week_log_select.index.to_pydatetime(), seasonal[-100:].values,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(cbn_usd_week_log_select.index.to_pydatetime(), residual[-100:].values, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
#Removing seasonality with differencing

cbn_usd_week_log_diff = cbn_usd_week_log - cbn_usd_week_log.shift()

plt.plot(cbn_usd_week_log_diff.index.to_pydatetime(), cbn_usd_week_log_diff.values)
cbn_usd_week_log_diff.dropna(inplace=True)

check_stationarity(cbn_usd_week_log_diff['Central Rate'])
#ACF and PACF plots



lag_auto_corr = acf(cbn_usd_week_log_diff['Central Rate'], nlags=10)

lag_par_auto_corr = pacf(cbn_usd_week_log_diff['Central Rate'], nlags=10, method='ols')



#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_auto_corr)

plt.axhline(y=0,linestyle='--',color='black')

plt.axhline(y=-1.96/np.sqrt(len(cbn_usd_week_log_diff['Central Rate'])),linestyle='--',color='black')

plt.axhline(y=1.96/np.sqrt(len(cbn_usd_week_log_diff['Central Rate'])),linestyle='--',color='black')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_par_auto_corr)

plt.axhline(y=0,linestyle='--',color='black')

plt.axhline(y=-1.96/np.sqrt(len(cbn_usd_week_log_diff['Central Rate'])),linestyle='--',color='black')

plt.axhline(y=1.96/np.sqrt(len(cbn_usd_week_log_diff['Central Rate'])),linestyle='--',color='black')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
model = ARIMA(cbn_usd_week_log['Central Rate'], order=(1, 1, 1))  

results_ARIMA = model.fit(disp=-1)
model = ARIMA(cbn_usd_week_log['Central Rate'], order=(1, 1, 1))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(cbn_usd_week_log_diff['Central Rate'].index.to_pydatetime(), cbn_usd_week_log_diff['Central Rate'].values)

plt.plot(cbn_usd_week_log_diff['Central Rate'].index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-cbn_usd_week_log_diff['Central Rate'])**2))
print(results_ARIMA.summary())

# plot residual errors

residuals = DataFrame(results_ARIMA.resid)

residuals.plot(kind='kde')

print(residuals.describe())
usd_predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print (usd_predictions_ARIMA_diff.head())
usd_predictions_ARIMA_diff_cumsum = usd_predictions_ARIMA_diff.cumsum()

usd_predictions_ARIMA_log = pd.Series(cbn_usd_week_log['Central Rate'].iloc[0], index=cbn_usd_week_log['Central Rate'].index)

usd_predictions_ARIMA_log = usd_predictions_ARIMA_log.add(usd_predictions_ARIMA_diff_cumsum,fill_value=0)
usd_predictions_ARIMA = np.exp(usd_predictions_ARIMA_log)

plt.plot(cbn_usd_week['Central Rate'].index.to_pydatetime(), cbn_usd_week['Central Rate'].values)

plt.plot(cbn_usd_week['Central Rate'].index.to_pydatetime(), usd_predictions_ARIMA.values)

plt.title('RMSE: %.4f'% np.sqrt(sum((usd_predictions_ARIMA-cbn_usd_week['Central Rate'])**2)/len(cbn_usd_week['Central Rate'])))
size = int(len(cbn_usd_week_log['Central Rate']) - 15)

train, test = cbn_usd_week_log['Central Rate'][0:size], cbn_usd_week_log['Central Rate'][size:len(cbn_usd_week_log['Central Rate'])]

historical = [x for x in train]

predictions = list()



print('Printing Predicted vs Expected Values...')

print('\n')

for t in range(len(test)):

    model = ARIMA(historical, order=(2,1,1))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(float(yhat))

    observed = test[t]

    historical.append(observed)

    print('Predicted USD Central Rate = %f, Expected USD Central Rate = %f' % (np.exp(yhat), np.exp(observed)))



error = mean_squared_error(test, predictions)



print('\n')

print('Printing Mean Squared Error of Predictions...')

print('Test MSE: %.6f' % error)



usd_predictions_series = pd.Series(predictions, index = test.index)
fig, ax = plt.subplots()

ax.set(title='Spot Exchange Rate', xlabel='Date', ylabel='USD Central Rates')

ax.plot(cbn_usd_week['Central Rate'][-50:], 'o', label='observed')

ax.plot(np.exp(usd_predictions_series), 'g', label='rolling one-step out-of-sample forecast')

legend = ax.legend(loc='upper left')

legend.get_frame().set_facecolor('w')