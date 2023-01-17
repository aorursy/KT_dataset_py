import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.api as sm

import warnings
passengers = pd.read_csv('../input/air-passengers/AirPassengers.csv')
passengers.head()
dates = pd.date_range(start='1949-01-01', freq='MS',periods=len(passengers))
dates
passengers['Month'] = dates.month

passengers['Year'] = dates.year
passengers.head()
passengers.dtypes
passengers.head()
import calendar

passengers['Month'] = passengers['Month'].apply(lambda x: calendar.month_abbr[x])

passengers.rename({'#Passengers':'Passengers'},axis=1,inplace=True)

passengers = passengers[['Month','Year','Passengers']]
passengers.head()
passengers.head()
passengers['Date'] = dates

passengers.set_index('Date',inplace=True)
passengers.head()
plt.figure(figsize=(10,8))

passengers.groupby('Year')['Passengers'].mean().plot(kind='bar')

plt.show()
print('From the above figure we can see that passengers are increasing with the increase in the year')
plt.figure(figsize=(10,8))

passengers.groupby('Month')['Passengers'].mean().reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).plot(kind='bar')

plt.show()
print('From the above figure we can see that more passengers can be seen between months June to September.')
passengers_count = passengers['Passengers']
plt.figure(figsize=(10,8))

passengers_count.plot()

plt.xlabel('Year')

plt.ylabel('Number of Passengers')

plt.show()
decompose = sm.tsa.seasonal_decompose(passengers_count,model='multiplicative',extrapolate_trend=8)
fig = decompose.plot()

fig.set_figheight(10)

fig.set_figwidth(8)

fig.suptitle('Decomposition of Time Series')
fig,axes = plt.subplots(2,2)

fig.set_figheight(10)

fig.set_figwidth(15)

axes[0][0].plot(passengers.index,passengers_count,label='Actual')

axes[0][0].plot(passengers.index,passengers_count.rolling(window=4).mean(),label='4 months rolling mean')

axes[0][0].set_xlabel('Year')

axes[0][0].set_ylabel('Number of Passengers')

axes[0][0].set_title('4 Months Rolling Mean')

axes[0][0].legend(loc='best')





axes[0][1].plot(passengers.index,passengers_count,label='Actual')

axes[0][1].plot(passengers.index,passengers_count.rolling(window=6).mean(),label='6 months rolling mean')

axes[0][1].set_xlabel('Year')

axes[0][1].set_ylabel('Number of Passengers')

axes[0][1].set_title('6 Months Rolling Mean')

axes[0][1].legend(loc='best')







axes[1][0].plot(passengers.index,passengers_count,label='Actual')

axes[1][0].plot(passengers.index,passengers_count.rolling(window=8).mean(),label='8 months rolling mean')

axes[1][0].set_xlabel('Year')

axes[1][0].set_ylabel('Number of Passengers')

axes[1][0].set_title('8 Months Rolling Mean')

axes[1][0].legend(loc='best')





axes[1][1].plot(passengers.index,passengers_count,label='Actual')

axes[1][1].plot(passengers.index,passengers_count.rolling(window=12).mean(),label='12 months rolling mean')

axes[1][1].set_xlabel('Year')

axes[1][1].set_ylabel('Number of Passengers')

axes[1][1].set_title('12 Months Rolling Mean')

axes[1][1].legend(loc='best')



plt.tight_layout()

plt.show()
passengers.head()
monthly = pd.pivot_table(data=passengers,values='Passengers',index='Month',columns='Year')

monthly = monthly.reindex(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
monthly
monthly.plot(figsize=(8,6))

plt.show()
yearly = pd.pivot_table(data=passengers,values='Passengers',index='Year',columns='Month')

yearly = yearly[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
yearly
yearly.plot(figsize=(8,6))

plt.show()
yearly.plot(kind='box',figsize=(8,6))

plt.show()
# Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller

adfuller(passengers_count)
adfuller_results = pd.Series(adfuller(passengers_count)[:4],index=['T stats','p-value','lags used','Number of observations'])

for key,value in adfuller(passengers_count)[4].items():

    adfuller_results['Critical Value'+' '+ key] = value

print(adfuller_results)
passengers_count.plot()

plt.show()
passengers_log = np.log10(passengers_count)
passengers_log.plot()

plt.show()
# Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller

adfuller(passengers_log)

adfuller_results = pd.Series(adfuller(passengers_log)[:4],index=['T stats','p-value','lags used','Number of observations'])

for key,value in adfuller(passengers_log)[4].items():

    adfuller_results['Critical Value (%s)'%key] = value

print(adfuller_results)
diff1 = passengers_count.diff(1)

diff1.head()
diff1.dropna(axis=0,inplace=True)
diff1.plot()

plt.show()
# Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller

adfuller(diff1)

adfuller_results = pd.Series(adfuller(diff1)[:4],index=['T stats','p-value','lags used','Number of observations'])

for key,value in adfuller(diff1)[4].items():

    adfuller_results['Critical Value (%s)'%key] = value

print(adfuller_results)
log_diff1 = passengers_log.diff(1)

log_diff1.head()
log_diff1.dropna(axis=0,inplace=True)
log_diff1.plot()

plt.show()
# Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller

adfuller(log_diff1)

adfuller_results = pd.Series(adfuller(log_diff1)[:4],index=['T stats','p-value','lags used','Number of observations'])

for key,value in adfuller(log_diff1)[4].items():

    adfuller_results['Critical Value (%s)'%key] = value

print(adfuller_results)
log_diff2 = passengers_log.diff(2)

log_diff2.head()
log_diff2.dropna(axis=0,inplace=True)
log_diff2.plot()

plt.show()
# Perform Dickey-Fuller test:

from statsmodels.tsa.stattools import adfuller

adfuller(log_diff2)

adfuller_results = pd.Series(adfuller(log_diff2)[:4],index=['T stats','p-value','lags used','Number of observations'])

for key,value in adfuller(log_diff2)[4].items():

    adfuller_results['Critical Value (%s)'%key] = value

print(adfuller_results)
import itertools

# Define the p, d and q parameters to take any value between 0 and 2

p = q = range(0, 3)

d = range(0,1)

# Generate all different combinations of p, d and q triplets

pdq = list(itertools.product(p, d, q))
pdq
# Generate all different combinations of seasonal p, q and q triplets

D = range(0,3)

P = Q = range(0, 3) 

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(P, D, Q))]
seasonal_pdq
import sys

warnings.filterwarnings("ignore") # specify to ignore warning messages



best_aic = np.inf

best_pdq = None

best_seasonal_pdq = None

temp_model = None



for param in pdq:

    for param_seasonal in seasonal_pdq:

       

        try:

            temp_model = sm.tsa.statespace.SARIMAX(log_diff2,

                                             order = param,

                                             seasonal_order = param_seasonal,

                                             enforce_stationarity=False,

                                             enforce_invertibility=False)

            results = temp_model.fit()



           # print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))

            if results.aic < best_aic:

                best_aic = results.aic

                best_pdq = param

                best_seasonal_pdq = param_seasonal

        except:

            #print("Unexpected error:", sys.exc_info()[0])

            continue

print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
sarima = sm.tsa.statespace.SARIMAX(log_diff2,order=(1,0,1),seasonal_order=(1,0,1,12),enforce_invertibility=False,enforce_stationarity=False)
sarima_results = sarima.fit()
print(sarima_results.summary())
passengers_count.tail(15)
prediction = sarima_results.get_prediction(start=pd.to_datetime('1960-01-01'),full_results=True)
prediction.predicted_mean
predicted_values = np.power(10,prediction.predicted_mean)
predicted_values
actual = passengers_count['1960-01-01':]
actual
# mean absolute percentage error

mape = np.mean(np.abs(actual - predicted_values)/actual)

mape
# mean square error

mse = np.mean((actual - predicted_values) ** 2)

mse
# Get forecast 36 steps (3 years) ahead in future

n_steps = 36

pred_uc_99 = sarima_results.get_forecast(steps=36, alpha=0.01) # alpha=0.01 signifies 99% confidence interval

pred_uc_95 = sarima_results.get_forecast(steps=36, alpha=0.05) # alpha=0.05 95% CI



# Get confidence intervals 95% & 99% of the forecasts

pred_ci_99 = pred_uc_99.conf_int()

pred_ci_95 = pred_uc_95.conf_int()


pred_ci_99.head()
pred_ci_95.head()
n_steps = 36

idx = pd.date_range(passengers_count.index[-1], periods=n_steps, freq='MS')

fc_95 = pd.DataFrame(np.column_stack([np.power(10, pred_uc_95.predicted_mean), np.power(10, pred_ci_95)]), 

                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])

fc_99 = pd.DataFrame(np.column_stack([np.power(10, pred_ci_99)]), 

                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_95.head()


fc_99.head()
fc_all = fc_95.combine_first(fc_99)

fc_all = fc_all[['forecast', 'lower_ci_95', 'upper_ci_95', 'lower_ci_99', 'upper_ci_99']] # just reordering columns

fc_all.head()
# plot the forecast along with the confidence band

axis = passengers_count.plot(label='Observed', figsize=(15, 6))

fc_all['forecast'].plot(ax=axis, label='Forecast', alpha=0.7)

#axis.fill_between(fc_all.index, fc_all['lower_ci_95'], fc_all['upper_ci_95'], color='k', alpha=.25)

axis.fill_between(fc_all.index, fc_all['lower_ci_99'], fc_all['upper_ci_99'], color='k', alpha=.25)

axis.set_xlabel('Years')

axis.set_ylabel('Tractor Sales')

plt.legend(loc='best')

plt.show()
sarima_results.plot_diagnostics(lags=30,figsize=(10,8))