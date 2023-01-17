# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
traffic_data = pd.read_csv(r'/kaggle/input/traffic-violations-in-maryland-county/Traffic_Violations.csv')
print(traffic_data.head())
# This is for violation count
forecast_violation = pd.DataFrame(traffic_data[['Date Of Stop']])
forecast_violation.columns = ['Date']
forecast_violation['No of violations'] = 1
# print(forecast_violation.head())

# Converting the Date into Datetime format
forecast_violation['Date'] = pd.to_datetime(forecast_violation.Date)
# print(forecast_violation.head())

# sorting the date and resetting the index
forecast_violation = forecast_violation.sort_values(by = 'Date')
forecast_violation = forecast_violation.reset_index(drop = True)

print(forecast_violation.head())
# Summing up the daily violations by month
forecast_violation_month = forecast_violation.resample('M', on = 'Date').sum()
print(forecast_violation_month.head())

# Creating a running sum of the violation of every month, this keeps the data on on an upward Trend
forecast_violation_cumsum = forecast_violation_month.cumsum()
print(forecast_violation_cumsum.head())

# forecast_fatal['Fatal'] = pd.Series(np.where(forecast_fatal.Fatal.values == 'Yes', 1, 0), forecast_fatal.index)
# forecast_belt['Belt'] = pd.Series(np.where(forecast_belt.Belt.values == 'Yes', 1, 0), forecast_belt.index)
# forecast_alcohol['Alcohol'] = pd.Series(np.where(forecast_alcohol.Alcohol.values == 'Yes', 1, 0), forecast_alcohol.index)
# Importing necessary files for the ARIMA forecasting

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# The monthly trend of the belt violations till now
forecast_violation_month.plot(figsize=(25, 6))
plt.show
# The daily trend of the belt violations till now
forecast_violation_cumsum.plot(figsize=(35, 6))
plt.show
# Define the p, d and q parameters to take any value between 0 and 2
p_m = d_m = q_m = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq_m = list(itertools.product(p_m, d_m, q_m))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq_m = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p_m, d_m, q_m))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq_m[1], seasonal_pdq_m[1]))
print('SARIMAX: {} x {}'.format(pdq_m[1], seasonal_pdq_m[2]))
print('SARIMAX: {} x {}'.format(pdq_m[2], seasonal_pdq_m[3]))
print('SARIMAX: {} x {}'.format(pdq_m[2], seasonal_pdq_m[4]))
# Finding parameters for the Seasonal ARIMA model, using grid search for hypterpameterization and model selection
# Grid searach is an iterative process of finding parameters, that inturn help to find optimal parameters 

warnings.filterwarnings("ignore") # specify to ignore warning messages

# This is to keep a track of minimum AIC value and record its parameters
min_value = 99999999
parameters = 0
sesonal_parameters = 0

for param in pdq_m:
    for param_seasonal in seasonal_pdq_m:
        try:
            mod_m = sm.tsa.statespace.SARIMAX(forecast_violation_cumsum,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results_m = mod_m.fit()
            
            if results_m.aic < min_value:
                min_value = results_m.aic
                parameters = param
                seasonal_parameters = param_seasonal
            

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results_m.aic))
        except:
            continue
print('################ Seasonal ARIMA Results ################')
print('Lowest AIC value: ', min_value)
print('Paramters: ',parameters)
print('Seasonal parameters: ', seasonal_parameters)
# Fitting ARIMA model for the violation data set created using parameters from the previous step
mod = sm.tsa.statespace.SARIMAX(forecast_violation_cumsum,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

# Capturing the result by fitting the model
results = mod.fit()

print(results.summary().tables[1])
# The Diagnostic of the ARIMA model
results.plot_diagnostics(figsize=(15, 12))
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2017-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = forecast_violation_cumsum['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7,figsize=(25, 6))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('# violations')
plt.legend()

plt.show()
y_forecasted = pd.DataFrame(pred.predicted_mean)

y_truth = forecast_violation_cumsum['2017-01-31':]
y_truth = y_truth.astype(float)


# Compute the absolute percentage error
from math import sqrt
mse = np.mean(np.abs((y_truth.values - y_forecasted.values) / y_truth.values)) * 100
print('The Mean Absolute Percentage Error of our forecasts is', round(mse, 2))

# COmpute Mean square error
# mse = ((y_forecasted.values - y_truth.values) ** 2).mean()
# print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# Future Predictions , for 24 months 
pred_uc = results.get_forecast(steps=24)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
# plot to show the future prediction upto 2 years
ax = forecast_violation_cumsum.plot(label='observed', figsize=(30, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('# violation')

plt.legend()
plt.show()
