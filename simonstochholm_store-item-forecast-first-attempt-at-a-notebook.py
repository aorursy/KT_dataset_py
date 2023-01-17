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




import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

#indlæs datasæt

df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv', index_col='date', parse_dates=True)
#kig på det

df.head()
sales_a = df[df.store == 2]
sales_a = sales_a[sales_a.item == 1]
sales = sales_a['sales']
sales
#plot data

sales.plot(grid=True)


from statsmodels.tsa.stattools import adfuller

# Run Dicky-Fuller test

result = adfuller(sales)



# Print test statistic

print(result[0])



# Print p-value

print(result[1])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create figure

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

 

# Plot the ACF of savings on ax1

plot_acf(sales, zero=False, ax=ax1, lags=10)



# Plot the PACF of savings on ax2

plot_pacf(sales, zero=False, ax=ax2, lags=10)



plt.show()
import warnings

warnings.simplefilter(action='ignore', category=Warning)
# Create empty list to store search results

order_aic_bic=[]



# Loop over p values from 0-2

for p in range(3):

  # Loop over q values from 0-2

    for q in range(3):

        try:

            # create and fit ARMA(p,q) model

            model = SARIMAX(sales, order=(p,0,q), seasonal_order=(1,1,0,7))

            results = model.fit()

            



            # Append order and results tuple

            order_aic_bic.append((p,q, results.aic, results.bic))

            print(p,q,results.aic, results.bic)

            

        except:

            print(p, q, None, None)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create and fit model

model = SARIMAX(sales, order=(2,0,1), trend='c')

results = model.fit()



# Create the 4 diagostics plots

results.plot_diagnostics()

plt.show()



# Print summary

print(results.summary())
# Import seasonal decompose

from statsmodels.tsa.seasonal import seasonal_decompose



# Perform additive decomposition

decomp = seasonal_decompose(sales, 

                            freq=7)



# Plot decomposition

decomp.plot()

plt.show()
# Take the first and seasonal differences and drop NaNs

sales_diff = sales.diff().diff(7).dropna()
# Create the figure 

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))



# Plot the ACF on ax1

plot_acf(sales_diff, zero=False, ax=ax1, lags=6)



# Plot the PACF on ax2

plot_pacf(sales_diff, zero=False, ax=ax2, lags=6)



plt.show()
# Make list of lags

lags = [7, 14, 21, 28, 35]



# Create the figure 

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))



# Plot the ACF on ax1

plot_acf(sales_diff, ax=ax1, lags=lags, zero=False)



# Plot the PACF on ax2

plot_pacf(sales_diff, ax=ax2, lags=lags, zero=False)



plt.show()
#!pip install pmdarima

import pmdarima as pm
# Create auto_arima model

model1 = pm.auto_arima(sales,

                      seasonal=True, m=7,

                      d=0, D=1, 

                 	  max_p=2, max_q=2,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True)

                       

# Print model summary

print(model1.summary())
# Import model class

from statsmodels.tsa.statespace.sarimax import SARIMAX



# Create model object

model = SARIMAX(sales, 

                order=(2,0,2), 

                seasonal_order=(2,1,1,7), 

                trend='c')

# Fit model

results = model.fit()
# Plot common diagnostics

results.plot_diagnostics()

plt.show()
# Create forecast object

forecast_object = results.get_forecast(steps=90)



# Extract prediction mean

mean = forecast_object.predicted_mean



# Extract the confidence intervals

conf_int = forecast_object.conf_int()



# Extract the forecast dates

dates = mean.index
plt.figure()



# Plot past CO2 levels

plt.plot(sales.index, sales, label='past')



# Plot the prediction means as line

plt.plot(dates, mean, label='predicted')



# Shade between the confidence intervals

plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.2)



# Plot legend and show figure

plt.legend()

plt.show()
# Print last predicted mean

print(mean.iloc[-1])



# Print last confidence interval

print(conf_int.iloc[-1])
mean
## Validating Forecast

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = sales['2014':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()
y_forecasted = pred.predicted_mean

y_truth = sales['2017-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#The MSE is a measure of the quality of an estimator — it is always non-negative, 

#and the smaller the MSE, the closer we are to finding the line of best fit.
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = sales.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()