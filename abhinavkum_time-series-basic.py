import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 5

import itertools

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/AirPassengers.csv')

data.head()
data.info()
data['Month'] =  pd.to_datetime(data['Month'], format='%Y-%m')
data.info()
data.set_index('Month', inplace=True)
data.head()
plt.plot(data)
from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries):

    

    #Determing rolling statistic

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries.unstack(), autolag='AIC') # this one only take series so unpack used.

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(data)
data_log = np.log(data)

plt.plot(data_log)
moving_avg = data_log.rolling(12).mean()

plt.plot(data_log)

plt.plot(moving_avg)
data_log_minus_moving_avg = data_log - moving_avg

data_log_minus_moving_avg.head(12)
data_log_minus_moving_avg.dropna(inplace=True)
test_stationarity(data_log_minus_moving_avg)
exponential_weight = data_log.ewm(halflife = 12).mean()

plt.plot(data_log)

plt.plot(exponential_weight, color='red')
data_log_minus_exponential_weight = data_log - exponential_weight

test_stationarity(data_log_minus_exponential_weight)
data_log_diff = data_log - data_log.shift()

plt.plot(data_log_diff)
data_log_diff.dropna(inplace=True)

test_stationarity(data_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

# decomposition = seasonal_decompose(data_log)

# trend = decomposition.trend

# seasonal = decomposition.seasonal

# residual = decomposition.resid





# plt.subplot(411)

# plt.plot(data_log, label='Original')

# plt.legend(loc='best')

# plt.subplot(412)

# plt.plot(trend, label='Trend')

# plt.legend(loc='best')

# plt.subplot(413)

# plt.plot(seasonal,label='Seasonality')

# plt.legend(loc='best')

# plt.subplot(414)

# plt.plot(residual, label='Residuals')

# plt.legend(loc='best')

# plt.tight_layout()

# rcParams['figure.figsize'] = 18, 8

decomposition = seasonal_decompose(data_log, model='additive')

fig = decomposition.plot()

plt.show()
data_log_decompose = decomposition.resid

data_log_decompose.dropna(inplace=True)

test_stationarity(data_log_decompose)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(data_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - data_log_diff.iloc[:,0])**2))
model = ARIMA(data_log, order=(2, 1, 2))  

results_AR = model.fit(disp=-1)  

plt.plot(data_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - data_log_diff.iloc[:,0])**2))
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)

predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_diff_cumsum.head()
predictions_ARIMA_log = pd.Series(data_log.iloc[0,0], index=data_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(data)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data.iloc[:,0])**2)/len(data)))
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
for param in pdq:

    for param_seasonal in seasonal_pdq:

#         try:

        mod = sm.tsa.statespace.SARIMAX(data,

                                        order=param,

                                        seasonal_order=param_seasonal,

                                        enforce_stationarity=False,

                                        enforce_invertibility=False)

        results = mod.fit()

        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

#         except:

#             continue
# ARIMA(0, 1, 1)x(1, 1, 1, 12)12 - AIC:920.3192974988726

mod = sm.tsa.statespace.SARIMAX(data,

                                order=(0, 1, 1),

                                seasonal_order=(1, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
start=pd.to_datetime('1953-01-01')

results.get_prediction(start).conf_int().head()
results.get_prediction(start=pd.to_datetime('1953-01-01'), dynamic=False).conf_int().head()
pred = results.get_prediction(start=pd.to_datetime('1953-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = data['1950':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()

y_forecasted = pred.predicted_mean.values

y_truth = data['1960-01-01':].values

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = data.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()