import pandas as pd

import datetime as dt

import numpy as np

from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

from patsy import dmatrices

from pandas import Series

from matplotlib import pyplot

from datetime import datetime

from matplotlib.pyplot import figure



import statsmodels.api as sm



figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')



from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt



import sys

import warnings

import itertools

warnings.filterwarnings("ignore")





import statsmodels.tsa.api as smt

import statsmodels.formula.api as smf



from statsmodels.tsa.stattools import adfuller





from statsmodels.tsa.arima_model import ARIMA
airpax_data = pd.read_csv("../input/AirPassengers.csv")

airpax_data.head()
#Parse strings to datetime type

airpax_data['Month'] = pd.to_datetime(airpax_data['Month'],infer_datetime_format=True) #convert from string to datetime

airpax_data = airpax_data.set_index(['Month'])
## plot graph

plt.xlabel('Date')

plt.ylabel('Number of air passengers')

plt.plot(airpax_data)
#Test whether Timeseries is Stationary or not

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12).mean()

    rolstd = timeseries.rolling(window=12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries['#Passengers'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(airpax_data)
airpax_df1 = airpax_data.diff(periods=1)

airpax_df1.dropna(inplace=True)

test_stationarity(airpax_df1)
airpax_log = np.log10(airpax_data)

airpax_log.dropna(inplace=True)

test_stationarity(airpax_log)
airpax_log_df = airpax_log.diff(periods=1)

airpax_log_df.dropna(inplace=True)

test_stationarity(airpax_log_df)
airpax_decompose = sm.tsa.seasonal_decompose(airpax_data['#Passengers'], model="multiplicative", freq=12)

airpax_decompose.plot()

plt.show()
trend = airpax_decompose.trend

seasonal = airpax_decompose.seasonal

residual = airpax_decompose.resid
print("Trend \n",trend.head(24))

print("Seasonal \n",seasonal.head(24))

print("Residual \n",residual.head(24))
train=airpax_data[0:int(len(airpax_data)*0.80)] 

test=airpax_data[int(len(airpax_data)*0.80):]

#Split the data for 21 month test



train_log = np.log10(train['#Passengers'])
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
model_airpax = ExponentialSmoothing(np.asarray(train['#Passengers']),seasonal_periods=24, trend='add',seasonal='mul').fit(optimized=True)
airpax_Holt = test.copy()
airpax_Holt['Holt']=model_airpax.forecast(len(test['#Passengers']))
mean_absolute_percentage_error(test['#Passengers'],airpax_Holt['Holt'])
model_airpax.params
airpax_Holt['Pax'] = model_airpax.forecast(len(test['#Passengers']))

plt.figure(figsize=(16,8))

plt.plot(train['#Passengers'], label='Train')

plt.plot(test['#Passengers'], label='Test')

plt.plot(airpax_Holt['Holt'], label='Holt Trend Add Seasonal Mul')

plt.legend(loc=0)
fig, axes = plt.subplots(1, 2)

fig.set_figwidth(12)

fig.set_figheight(4)

smt.graphics.plot_acf(airpax_log_df, lags=20, ax=axes[0])

smt.graphics.plot_pacf(airpax_log_df, lags=20, ax=axes[1])

plt.tight_layout()
# Define the p, d and q parameters to take any value between 0 and 2

p = d = q = range(0, 2)



# Generate all different combinations of p, d and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
best_aic = np.inf

best_pdq = None

best_seasonal_pdq = None

temp_model = None

train_log = np.log10(train['#Passengers'])
for param in pdq: #Non-Seasonal

    for param_seasonal in seasonal_pdq:

        

        try:

            temp_model = sm.tsa.statespace.SARIMAX(train_log,

                                             order = param,

                                             seasonal_order = param_seasonal)

            results = temp_model.fit()



            # print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))

            if results.aic < best_aic:

                best_aic = results.aic

                best_pdq = param

                best_seasonal_pdq = param_seasonal #Store the best param values : AIC, P, D, Q [Non-S / Seas]

        except:

            #print("Unexpected error:", sys.exc_info()[0])

            continue

print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
mod = sm.tsa.statespace.SARIMAX(train_log,

                                order=(0,1,1),

                                seasonal_order=(1,0,1,12),

                                enforce_stationarity=True)



best_results = mod.fit()



print(best_results.summary().tables[1])
best_results.summary().tables[1]
pred_dynamic = best_results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=True, full_results=True)

pred_dynamic_ci = pred_dynamic.conf_int()
pred99 = best_results.get_forecast(steps=24, alpha=0.1)
# Extract the predicted and true values of our time series

sales_ts_forecasted = pred_dynamic.predicted_mean

testCopy = test.copy()

testCopy['Passengers_Forecast'] = np.power(10, pred99.predicted_mean)
# Compute the root mean square error

mse = ((testCopy['#Passengers'] - testCopy['Passengers_Forecast']) ** 2).mean()

rmse = np.sqrt(mse)

print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 3)))
axis = train['#Passengers'].plot(label='Train Sales', figsize=(10, 6))

testCopy['#Passengers'].plot(ax=axis, label='Test Sales', alpha=0.7)

testCopy['Passengers_Forecast'].plot(ax=axis, label='Forecasted ', alpha=0.7)

axis.set_xlabel('Years')

axis.set_ylabel('Passengers')

plt.legend(loc='best')

plt.show()

plt.close()
# Get forecast 36 steps (3 years) ahead in future

n_steps = 36

pred_uc_99 = best_results.get_forecast(steps=36, alpha=0.01) # alpha=0.01 signifies 99% confidence interval

pred_uc_95 = best_results.get_forecast(steps=36, alpha=0.05) # alpha=0.05 95% CI



# Get confidence intervals 95% & 99% of the forecasts

pred_ci_99 = pred_uc_99.conf_int()

pred_ci_95 = pred_uc_95.conf_int()
n_steps = 36

idx = pd.date_range(airpax_data.index[-1], periods=n_steps, freq='MS')

fc_95 = pd.DataFrame(np.column_stack([np.power(10, pred_uc_95.predicted_mean), np.power(10, pred_ci_95)]), 

                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])

fc_99 = pd.DataFrame(np.column_stack([np.power(10, pred_ci_99)]), 

                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])

fc_all = fc_95.combine_first(fc_99)

fc_all = fc_all[['forecast', 'lower_ci_95', 'upper_ci_95', 'lower_ci_99', 'upper_ci_99']] # just reordering columns

fc_all.head()
# plot the forecast along with the confidence band

axis = airpax_data['#Passengers'].plot(label='Observed', figsize=(8, 4))

fc_all['forecast'].plot(ax=axis, label='Forecast', alpha=0.7)

axis.fill_between(fc_all.index, fc_all['lower_ci_95'], fc_all['upper_ci_95'], color='k', alpha=.15)

axis.set_xlabel('Years')

axis.set_ylabel('Tractor Sales')

plt.legend(loc='best')

plt.show()
best_results.plot_diagnostics(lags=30, figsize=(16,12))

plt.show()