import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Load in the time series

candy = pd.read_csv('/kaggle/input/week6dataset/candy_production.csv', 

                 index_col='date', 

                 parse_dates=True)



# Plot the time series

fig, ax = plt.subplots()

candy.plot(ax=ax)

plt.show()
# Split the data into a train and test set

candy_train = candy.loc[:'2006']

candy_test = candy.loc['2007':]



# Create an axis

fig, ax = plt.subplots()



# Plot the train and test setsa dn show

candy_train.plot(ax=ax)

candy_test.plot(ax=ax)

plt.show()
earthquake = pd.read_csv('/kaggle/input/week6dataset/earthquakes.csv', parse_dates = ['date'], index_col = 'date').drop('Year', axis = 1)

earthquake.head()
# Import augmented dicky-fuller test function

from statsmodels.tsa.stattools import adfuller



# Run test

result = adfuller(earthquake['earthquakes_per_year'])



# Print test statistic

print(result[0])



# Print p-value

print(result[1])



# Print critical values

print(result[4]) 
city = pd.DataFrame([1.0, 0.96, 0.96, 0.95, 0.99, 0.97, 1.02, 1.0, 1.01, 1.01, 1.04, 0.99, 1.02, 1.03, 1.05, 1.07, 1.1, 1.07, 1.06, 1.08, 1.11, 1.1, 1.14, 1.08, 1.16, 1.15, 1.15, 1.19, 1.22, 1.18, 1.2, 1.2, 1.21, 1.24, 1.27, 1.27, 1.28, 1.31, 1.34, 1.35, 1.33, 1.36, 1.39, 1.41, 1.39, 1.44, 1.42, 1.45, 1.43, 1.46, 1.47, 1.51, 1.5, 1.58, 1.58, 1.54, 1.57, 1.62, 1.62, 1.65, 1.65, 1.66, 1.71, 1.71, 1.73, 1.76, 1.77, 1.81, 1.83, 1.86, 1.91, 1.88, 1.92, 1.94, 1.9, 1.99, 1.98, 2.03, 2.02, 2.03, 2.1, 2.13, 2.1, 2.14, 2.16, 2.19, 2.18, 2.23, 2.24, 2.29, 2.32, 2.31, 2.37, 2.39, 2.42, 2.44, 2.48, 2.55, 2.52, 2.56]

, columns = ['city_population'])
# Run the ADF test on the time series

result = adfuller(city['city_population'])



# Plot the time series

fig, ax = plt.subplots()

city.plot(ax=ax)

plt.show()



# Print the test statistic and the p-value

print('ADF Statistic:', result[0])

print('p-value:', result[1])
# Calculate the first difference of the time series

city_stationary = city.diff().dropna()



# Run ADF test on the differenced time series

result = adfuller(city_stationary['city_population'])



# Plot the differenced time series

fig, ax = plt.subplots()

city_stationary.plot(ax=ax)

plt.show()



# Print the test statistic and the p-value

print('ADF Statistic:', result[0])

print('p-value:', result[1])
# Calculate the second difference of the time series

city_stationary = city.diff().diff().dropna()



# Run ADF test on the differenced time series

result = adfuller(city_stationary['city_population'])



# Plot the differenced time series

fig, ax = plt.subplots()

city_stationary.plot(ax=ax)

plt.show()



# Print the test statistic and the p-value

print('ADF Statistic:', result[0])

print('p-value:', result[1])
amazon = pd.read_csv('/kaggle/input/week6dataset/amazon_close.csv', parse_dates = ['date'], index_col = 'date')

amazon.head()
# Calculate the first difference and drop the nans

amazon_diff = amazon.diff().dropna()



# Run test and print

result_diff = adfuller(amazon_diff['close'])

print(result_diff)



# Calculate log-return and drop nans

amazon_log = np.log((amazon/amazon.shift(1)).dropna())



# Run test and print

result_log = adfuller(amazon_log['close'])

print(result_log)
# Import data generation function and set random seed

from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(1)



# Set coefficients

ar_coefs = [1,]

ma_coefs = [1, -0.7]



# Generate data

y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, sigma=0.5, )



plt.plot(y)

plt.ylabel(r'$y_t$')

plt.xlabel(r'$t$')

plt.show()
# Import data generation function and set random seed

from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(2)



# Set coefficients

ar_coefs = [1, -0.3, -0.2]

ma_coefs = [1,]



# Generate data

y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, sigma=0.5, )



plt.plot(y)

plt.ylabel(r'$y_t$')

plt.xlabel(r'$t$')

plt.show()
# Import data generation function and set random seed

from statsmodels.tsa.arima_process import arma_generate_sample

np.random.seed(3)



# Set coefficients

ar_coefs = [1, 0.2]

ma_coefs = [1, 0.3, 0.4]



# Generate data

y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, sigma=0.5, )



plt.plot(y)

plt.ylabel(r'$y_t$')

plt.xlabel(r'$t$')

plt.show()
# Import the ARMA model

from statsmodels.tsa.arima_model import ARMA



# Instantiate the model

model = ARMA(y, order=(1,1))



# Fit the model

results = model.fit()
sample = pd.DataFrame([-0.18, -0.25, -0.26, -0.28, -0.38, -0.01, 0.16, 0.3, 0.16, 0.1, 0.23, -0.06, 0.1, 0.2, -0.03, 0.09, -0.12, -0.3, 0.23, 0.3, 0.39, 0.1, -0.07, 0.04, -0.04, 0.42, 0.43, 0.57, 0.31, 0.12, -0.01, -0.29, -0.3, -0.2, -0.15, -0.25, -0.49, -0.24, -0.11, 0.28, 0.08, -0.05, -0.34, -0.23, 0.12, 0.15, 0.02, 0.02, 0.25, 0.32, -0.03, -0.54, -0.4, 0.03, -0.03, -0.23, -0.32, -0.17, -0.26, -0.5, -0.24, -0.2, -0.24, 0.03, 0.21, 0.25, 0.05, -0.27, 0.02, 0.5, 0.59, -0.12, -0.14, -0.02, 0.14, 0.13, -0.13, -0.37, 0.07, 0.12, 0.1, 0.28, 0.07, -0.27, -0.25, 0.04, -0.23, -0.51, -0.65, -0.17, 0.29, 0.69, 0.39, 0.38, 0.56, 0.27, 0.1, -0.09, 0.15, 0.09], columns = ['timeseries_1'])
# Instantiate the model

model = ARMA(sample['timeseries_1'], order=(2,0))



# Fit the model

results = model.fit()



# Print summary

print(results.summary())
sample['timeseries_2'] = [-0.18, -0.12, -0.22, -0.17, -0.28, 0.15, -0.01, 0.33, -0.02, 0.12, 0.14, -0.16, 0.28, -0.02, -0.0, 0.17, -0.29, -0.08, 0.28, 0.02, 0.48, -0.17, 0.03, 0.01, -0.08, 0.53, 0.06, 0.61, -0.03, 0.18, -0.1, -0.23, -0.11, -0.16, -0.07, -0.19, -0.38, -0.04, -0.18, 0.4, -0.17, 0.1, -0.43, -0.0, 0.08, 0.08, 0.05, -0.01, 0.2, 0.18, -0.09, -0.43, -0.16, 0.02, -0.05, -0.1, -0.29, -0.07, -0.24, -0.32, -0.06, -0.28, -0.05, 0.06, 0.08, 0.21, -0.05, -0.23, 0.16, 0.33, 0.41, -0.25, 0.09, -0.21, 0.27, 0.05, -0.15, -0.26, 0.19, -0.09, 0.24, 0.16, -0.09, -0.14, -0.15, 0.05, -0.27, -0.26, -0.53, 0.06, 0.18, 0.6, 0.06, 0.41, 0.26, 0.1, 0.18, -0.2, 0.27, -0.08]
# Instantiate the model

model = ARMA(sample['timeseries_2'], order=(0,3))



# Fit the model

results = model.fit()



# Print summary

print(results.summary())
import warnings

warnings.filterwarnings('ignore')

# Instantiate the model

model = ARMA(earthquake, (3, 1))



# Fit the model

results = model.fit()



# Print model fit summary

print(results.summary())
hospital = pd.DataFrame([1.75, 1.66, 1.65, 1.62, 1.48, 1.77, 1.99, 2.17, 1.57, 1.28, 1.44, 1.06, 1.06, 1.2, 0.89, 1.04, 0.77, 0.54, 1.23, 1.33, 1.65, 1.27, 1.26, 1.4, 1.51, 2.13, 2.35, 2.53, 2.19, 1.72, 1.55, 1.19, 0.96, 1.1, 1.16, 1.03, 0.71, 0.82, 1.0, 1.51, 1.25, 1.07, 0.69, 1.26, 1.73, 1.76, 1.6, 1.59, 2.32, 2.41, 1.95, 1.06, 1.24, 1.61, 1.53, 1.26, 0.72, 0.71, 0.59, 0.26, 0.61, 0.66, 0.61, 0.97, 1.2, 1.26, 1.0, 0.58, 1.17, 1.81, 2.13, 1.19, 1.38, 1.54, 1.75, 1.74, 1.39, 0.87, 1.66, 1.72, 1.48, 1.73, 1.45, 1.0, 1.23, 1.4, 1.05, 0.67, 0.5, 1.13, 1.74, 2.69, 2.29, 2.28, 2.52, 1.92, 1.91, 1.66, 1.98, 1.9, 1.4, 1.01, 1.21, 1.46, 1.8, 1.3, 1.02, 1.46, 1.6, 1.63, 1.47, 1.37, 1.22, 1.38, 1.6, 2.44, 2.45, 2.02, 1.72, 1.49, 1.4, 1.32, 1.69, 2.01, 2.24, 1.86, 1.4, 1.67, 2.14, 1.51, 1.09, 1.24, 1.66, 1.28, 0.99, 1.15, 1.28, 0.96, 1.3, 1.28, 1.71, 1.56, 1.17, 1.36, 1.78, 2.08, 1.97, 2.0, 1.97, 2.02, 1.59, 1.21, 0.86, 0.19, 0.76, 1.08, 0.8, 0.57, 0.94, 1.37, 1.61, 1.96, 1.56, 1.1, 1.6, 1.71, 1.29, 1.55], columns = ['wait_times_hrs'])

hospital['nurse_count'] = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 7.0, 9.0, 9.0, 9.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 9.0, 9.0, 7.0, 7.0, 5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 5.0, 9.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 9.0, 9.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 5.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 3.0]
# Instantiate the model

model = ARMA(hospital['wait_times_hrs'], (2, 1), exog = hospital['nurse_count'])



# Fit the model

results = model.fit()



# Print model fit summary

print(results.summary())
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Instantiate the model

model = SARIMAX(amazon['close'], order = (3,1,3), seasonal_order = (1, 0, 1, 7))



# Fit the model

results = model.fit()



# Print model fit summary

print(results.summary())

# Generate predictions

mean_forecast = results.predict(start=80)



# Get confidence intervals of predictions

confidence_intervals = results.conf_int()



# # Print best estimate predictions

display(mean_forecast.head(), confidence_intervals)
# plot the amazon data

plt.plot(amazon.index, amazon, label='observed')



# plot your mean predictions

plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')



# shade the area between your confidence limits

# plt.fill_between(lower_limits.index, lower_limits,

# 		upper_limits, color='pink')



# set labels, legends and show plot

plt.xlabel('Date')

plt.ylabel('Amazon Stock Price - Close USD')

plt.legend()

plt.show()
# Generate predictions

mean_forecast = results.predict(start = 80, dynamic = True)



# Print best estimate predictions

display(mean_forecast.head())
# plot the amazon data

plt.plot(amazon.index, amazon, label='observed')



# plot your mean forecast

plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')



# shade the area between your confidence limits

# plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')



# set labels, legends and show plot

plt.xlabel('Date')

plt.ylabel('Amazon Stock Price - Close USD')

plt.legend()

plt.show()
# Take the first difference of the data

amazon_diff = amazon.diff().dropna()



# Create ARMA(2,2) model

arma = SARIMAX(amazon_diff, order=(2,0,2))



# Fit model

arma_results = arma.fit()



# Print fit summary

print(arma_results.summary())
# Make arma forecast of next 10 differences

arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean



# Integrate the difference forecast

arma_int_forecast = np.cumsum(arma_diff_forecast)



# Make absolute value forecast

arma_value_forecast = arma_int_forecast + amazon.iloc[-1,0]



# Print forecast

print(arma_value_forecast)
# Create ARIMA(2,1,2) model

arima = SARIMAX(amazon, order=(2,1,2))



# Fit ARIMA model

arima_results = arima.fit()



# Make ARIMA forecast of next 10 values

arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean



# Print forecast

print(arima_value_forecast)
df = pd.DataFrame([1.62, -0.94, 0.08, -0.66, 0.74, -2.95, 2.14, -1.54, 0.3, -0.02, 1.38, -2.33, 0.45, -0.5, 0.5, -1.54, 0.27, -0.83, -0.16, 0.26, -1.47, 1.55, 0.52, 0.34, 1.41, -0.44, 0.43, -0.85, -0.32, 0.27, -1.16, -0.18, -0.66, -1.03, -0.83, -0.34, -1.57, 0.25, 1.27, 0.15, 0.23, -0.13, -0.4, 1.52, -0.78, -0.36, 0.84, 1.89, -0.43, 1.28, 0.84, -0.19, -0.8, -0.14, -0.59, 0.18, 0.55, 0.88, 0.53, 1.36, -0.57, 1.75, 0.3, -0.25, 1.08, -0.11, 1.2, 1.42, 2.2, -1.04, -0.05, 0.02, -0.59, 0.26, 0.04, -1.77, 0.46, 0.38, -0.63, 0.87, -0.06, 0.14, 0.39, 0.25, 0.11, 0.26, -0.51, 0.61, -0.12, 1.02, 1.12, 0.32, 0.29, -0.15, 0.49, -0.31, -0.42, 0.26, -0.71, 0.73, -0.76, 1.34, 0.23, 0.75, -0.73, 0.69, 0.56, -1.38, 0.2, 0.02, -1.75, 0.52, 0.38, -1.35, 0.87, -1.39, 0.07, -1.9, 1.04, -0.31, -0.25, -0.31, 1.54, 1.47, -2.1, 2.58, 1.41, -0.17, -0.41, 1.69, -0.61, -0.67, -0.9, 0.56, 0.13, -0.99, 1.05, -1.2, 1.0, -0.3, -0.3, 0.19, 0.85, 0.49, 0.61, 0.52, 0.43, 0.8, 0.17, 0.84, -0.19, -2.1, 1.64, 1.16, -0.41, 0.78, 0.67, 0.01, -0.03, -1.2, -0.32, -1.22, -0.04, -0.8, 0.33, -0.29, 1.08, 0.11, 2.39, -1.97, 0.45, 1.12, 1.59, -0.68, 1.12, 0.45, 1.32, -0.61, 1.07, -0.14, -1.09, 0.68, -0.04, 0.9, -0.12, -0.06, 1.04, -0.09, 0.22, 0.11, 1.33, 0.2, 2.16, 1.19, 1.12, -0.85, 1.46, 0.01, 0.42, 1.19, -0.24, 1.46, -0.53, 2.19, -1.64, -1.08, -1.11, -1.49, -1.96, 1.42, -1.32, -1.03, 1.87, -1.31, -1.19, 1.28, -0.1, -0.67, 1.94, -0.1, 1.88, 0.64, 1.98, 0.43, -0.62, 3.19, -1.37, 0.28, -0.09, -0.29, 0.01, -0.12, 0.63, 1.79, -0.22, -0.47, -0.43, -0.84, -2.27, 0.61, -1.37, -0.47, 0.16, -1.64, -0.68, 0.43, -0.19, -0.62, 2.29, -1.62, 0.33, -0.01, -3.24, 2.32, -0.93, -1.37, 2.95, -0.03, 0.61, 0.1, 0.53, -1.6, 0.95, -0.96, -0.01, 1.3, -0.03, 0.61, 0.67, -0.62, 1.14, 1.24, 0.48, 1.45, -0.36, -0.2, -1.27, 1.0, -1.19, 1.39, -0.77, 0.06, 0.37, 0.54, -0.23, 0.5, 0.27, 0.26, -1.08, 0.36, -0.48, -0.33, 0.47, -0.85, 0.18, -0.34, 0.25, 0.59, -1.08, 1.14, 1.9, -2.0, 0.26, -0.54, -0.79, 1.17, -0.66, -0.29, 1.25, -0.93, 1.22, 0.09, 1.48, -0.51, 3.67, 0.44, 1.2, 0.97, -0.27, -0.54, -0.2, -1.25, 0.14, -1.16, 0.63, 0.67, -0.96, 1.81, -0.4, 1.35, 0.49, 1.31, -0.74, 0.56, 1.51, -1.24, -1.51, -1.29, -2.91, -2.81, -1.32, -0.03, -1.46, 0.67, 0.08, -0.01, -0.11, -0.17, 0.45, -1.37, 1.33, 0.5, -0.43, 1.42, -0.64, -0.21, 0.29, -0.59, -0.52, -0.59, -0.61, 0.71, -2.5, 0.86, -0.61, 0.3, 0.4, -2.74, 1.51, -0.47, -0.7, -0.2, 1.05, -0.72, 2.03, 0.33, 2.45, -0.49, -0.79, 1.47, -0.05, -0.45, -1.08, 0.13, 0.26, -0.9, -0.03, -0.02, -0.99, -0.43, -1.65, 1.61, -1.42, 0.68, -0.73, 1.0, -2.2, 1.23, -0.54, 1.83, -1.46, 0.96, 0.59, 0.79, -0.54, 0.65, -1.92, -0.5, 0.0, 0.12, -0.62, 2.58, 1.6, 0.12, -1.26, -0.78, -1.16, 0.01, -0.68, -2.05, 0.62, 1.12, -1.39, -0.26, 2.58, -0.69, -0.1, 1.08, -0.18, -0.47, 1.4, 0.75, 0.85, 0.43, 1.75, -0.06, 0.79, 1.27, 0.73, 2.08, 0.65, -0.99, -0.15, 1.21, -1.45, 0.51, -0.62, 1.19, -0.72, 1.58, -0.64, 1.55, 0.73, -0.68, 1.78, -1.14, -0.53, 0.95, -1.29, -0.5, -0.2, -1.3, 0.96, -0.0, -0.18, 0.4, 0.63, 1.62, 0.62, -0.02, 1.67, -0.17, -0.09, 2.1, -0.89, 0.1, -1.42, -1.28, 0.0, -1.79, 0.37, -1.29, -0.6, 0.65, 0.76, 0.27, 1.72, 2.08, 1.24, -0.26, 0.9, 0.69, -0.85, 0.69, -1.6, -0.5, 0.36, -3.07, -0.75, -0.31, -2.72, 0.06, -1.33, -0.63, -0.42, -0.09, -0.77, 0.18, -0.73, 0.53, 1.43, -0.16, 0.81, -0.63, 1.04, -1.19, -0.8, 0.7, -0.4, 1.0, 1.44, -0.44, 1.0, 2.01, 0.49, 1.09, 1.98, -0.36, 1.36, 1.49, 1.35, -0.52, 0.82, 0.68, -0.68, -0.29, 0.79, -0.69, -1.63, 0.82, -1.14, -1.07, -1.44, 0.98, -3.12, -1.42, -0.3, 0.11, 0.47, -1.13, 0.81, -1.43, 0.91, -2.19, 0.0, 0.92, -0.91, 1.78, -0.92, 0.92, 0.96, -0.46, 1.78, 0.48, -1.37, 4.87, -0.47, 0.67, -0.04, 0.59, -0.86, 0.34, 0.13, 1.83, 0.08, 1.15, 0.87, -0.03, 0.11, 1.2, 0.46, -0.51, -0.42, 0.2, 0.65, -0.61, 2.08, -0.78, 0.81, -2.25, -0.28, -0.9, -0.36, 1.34, -0.25, 2.11, 0.89, 1.3, -0.03, 1.97, -1.19, 0.99, -0.96, -0.5, -1.49, -0.67, -0.14, -0.73, -1.38, 1.82, -0.38, 0.98, 1.13, -0.42, -1.27, -0.49, 0.63, -2.16, 1.12, -0.72, 0.44, -1.84, 1.47, -0.7, 0.7, 0.35, 1.01, 0.06, 1.06, -1.45, -0.59, 0.71, -0.96, -0.33, 0.07, 0.46, -0.82, 0.73, -1.05, -0.25, 0.94, 0.48, -1.14, 1.67, -1.21, 0.86, -2.11, -0.83, -0.64, -0.97, 0.97, 0.42, 1.77, 1.59, 0.22, 0.62, 1.4, -0.34, 0.08, -0.97, 0.13, -3.22, -0.33, -1.76, -0.09, -1.05, -0.32, 1.48, -1.56, 0.83, -1.87, 0.25, -0.08, 0.24, -0.92, 0.88, 0.24, -0.26, 0.35, 0.52, 0.34, -0.89, -0.74, 2.57, 0.76, -1.38, 2.06, 0.7, 0.34, 0.31, -0.54, 0.27, -0.69, -0.1, -0.5, -0.19, 1.0, 0.91, -1.16, 0.45, -0.68, 0.89, -0.68, 1.59, -0.42, 0.54, -1.06, 1.44, -1.02, -1.43, -0.01, -2.74, -0.19, -0.75, -0.44, 1.46, -0.48, 1.4, 0.51, -0.1, -1.7, 1.4, -1.37, -0.22, -1.14, 0.39, -0.32, -0.29, -0.02, 1.12, 0.72, 0.81, 3.78, -0.76, 0.95, -0.12, 0.35, -1.19, -2.73, -0.75, -0.7, -0.18, -2.1, 0.02, -0.18, -1.35, 0.45, -0.36, 0.26, -0.63, 1.78, -0.43, 1.54, -0.24, 1.07, -0.54, -0.77, -0.54, -0.39, -0.71, 1.36, -1.54, -0.46, 2.04, -0.4, -1.82, 1.57, -0.24, -2.61, 0.28, -0.52, -0.02, 0.69, 0.12, 1.52, -0.04, -0.49, -0.5, 0.43, -2.6, 0.95, -1.67, -0.52, 0.45, 0.68, -2.14, 0.1, -0.01, -1.43, 0.75, -2.42, 0.86, -0.01, -0.55, 1.33, -1.31, 1.7, -0.96, 0.83, -0.35, 0.89, 0.74, -1.47, 0.62, -0.27, -0.7, -0.78, 1.9, -0.46, 1.84, -0.26, -0.42, 0.64, -1.08, 2.22, -0.36, 1.65, 0.34, 0.14, 0.31, 0.2, 0.67, -0.19, -0.29, -0.71, -0.1, 0.06, -0.0, 0.04, -0.29, 0.4, 0.36, -0.29, 0.34, -0.01, 0.97, -0.63, 0.5, 0.5, 0.57, 1.52, 0.71, 1.13, -0.94, 1.28, 1.91, -1.55, 0.02, 0.43, 0.51, -0.05, -0.34, 0.52, 1.06, 0.31, 2.59, 1.03, 1.08, 1.57, -0.06, -0.62, 0.45, 0.06, -1.74, -0.04, -0.54, -0.24, -0.58, 0.47, 0.01, 1.1, -1.76, 2.35, 0.89, -0.25, 1.56, 0.92, -0.65, 0.69, 0.71, -0.02, 1.27, 0.21, 0.47, -0.44, -0.16, -0.59, 1.46, -0.35, -0.67, 1.18, -0.5, -3.2, 1.15, -2.62, 0.12, -1.4, -1.56, 0.67, -2.32, -0.44, -0.01, -0.01, 0.55, 0.06, -0.42, 2.3, -1.25, 2.25, -0.42, 1.61, 1.24, -1.05, 1.23, 0.69, 0.34, -0.01, 0.78, -0.3, -1.06, 1.19, -1.23, 1.44, 0.45, -0.77, 0.61, -0.34, -1.13, 0.54, -0.93, 1.62, 0.54, -0.71, 0.9, -0.23, -1.79, 1.17, 0.23, 0.55, -0.08, 1.72, -0.28, 1.16, 1.88, -0.0, -0.62, 0.55, 0.32, 0.57, -1.28, 0.59, 0.3, -0.01, 0.84, -0.93, 0.17, 1.08, -0.22, 0.55, 1.46, 0.13, 0.98, -0.05, -1.09, -0.96, -0.16, -1.01, -0.98, 0.41, -1.21, 0.19, -0.21, -1.07, -2.1, 1.66, -0.92, -2.98, 0.64, -0.23, -1.23, 0.62, -1.04, 0.48, -0.21, -0.41, -0.15, 0.17, 0.77, -0.72, 0.04, 0.36, -0.43, 0.34, 0.65, 1.23, -0.04, -1.65, 0.74, -0.35, -0.96], columns = ['y'])
# Import

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# Create figure

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

 

# Plot the ACF of df

plot_acf(df, lags=10, zero=False, ax=ax1)



# Plot the PACF of df

plot_pacf(df, lags=10, zero=False, ax=ax2)



plt.show()
# Create figure

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))



# Plot ACF and PACF

plot_acf(earthquake, lags=10, zero=False, ax=ax1)

plot_pacf(earthquake, lags=10, zero=False, ax=ax2)



# Show plot

plt.show()



# Instantiate model

model = SARIMAX(earthquake, order =(1,0,0))



# Train model

results = model.fit()
# Create empty list to store search results

order_aic_bic=[]



# Loop over p values from 0-2

for p in range(3):

  # Loop over q values from 0-2

    for q in range(3):

      	# create and fit ARMA(p,q) model

        model = SARIMAX(df, order=(p,0,q))

        results = model.fit()

        

        # Append order and results tuple

        order_aic_bic.append((p,q,results.aic, results.bic))
# Construct DataFrame from order_aic_bic

order_df = pd.DataFrame(order_aic_bic, 

                        columns=['p', 'q', 'AIC', 'BIC'])



# Print order_df in order of increasing AIC

display(order_df.sort_values('AIC'))



# Print order_df in order of increasing BIC

display(order_df.sort_values('BIC'))
# Loop over p values from 0-2

for p in range(3):

    # Loop over q values from 0-2

    for q in range(3):

      

        try:

            # create and fit ARMA(p,q) model

            model = SARIMAX(earthquake, order=(p,0,q))

            results = model.fit()

            

            # Print order and results

            print(p, q, results.aic, results.bic)

            

        except:

            print(p, q, None, None)     
# Fit model

model = SARIMAX(earthquake, order=(1,0,1))

results = model.fit()



# Calculate the mean absolute error from residuals

mae = np.mean(np.abs(results.resid))



# Print mean absolute error

print(mae)



# Make plot of time series for comparison

earthquake.plot()

plt.show()
# Create and fit model

model1 = SARIMAX(df, order=(3, 0, 1))

results1 = model1.fit()



# Print summary

print(results1.summary())
# Create and fit model

model2 = SARIMAX(df, order=(2,0,0))

results2 = model2.fit()



# Print summary

print(results2.summary())
# Create and fit model

model = SARIMAX(df, order=((1, 1, 1)))

results=model.fit()



# Create the 4 diagostics plots

results.plot_diagnostics()

plt.show()
savings = pd.DataFrame([4.9, 5.2, 5.7, 5.7, 6.2, 6.7, 6.9, 7.1, 6.6, 7.0, 6.9, 6.4, 6.6, 6.4, 7.0, 7.3, 6.0, 6.3, 4.8, 5.3, 5.4, 4.7, 4.9, 4.4, 5.1, 5.3, 6.0, 5.9, 5.9, 5.6, 5.3, 4.5, 4.7, 4.6, 4.3, 5.0, 5.2, 6.2, 5.8, 6.7, 5.7, 6.1, 7.2, 6.5, 6.1, 6.3, 6.4, 7.0, 7.6, 7.2, 7.5, 7.0, 7.6, 7.2, 7.5, 7.8, 7.2, 7.5, 5.6, 5.7, 4.9, 5.1, 6.2, 6.0, 6.1, 7.5, 7.8, 8.0, 8.0, 8.1, 7.6, 7.1, 6.6, 5.6, 5.9, 6.6, 6.8, 7.8, 7.9, 8.7, 7.7, 7.3, 6.7, 7.5, 6.4, 9.7, 7.5, 7.1, 6.4, 6.0, 5.7, 5.0, 4.2, 5.1, 5.4, 5.1, 5.3, 5.0, 4.8, 4.7, 5.0, 5.4], columns = ['savings'])
# Plot time series

savings.plot()

plt.show()



# Run Dicky-Fuller test

result = adfuller(savings['savings'])



# Print test statistic

print(result[0])



# Print p-value

print(result[1])
# Create figure

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

 

# Plot the ACF of savings on ax1

plot_acf(savings, lags=10, zero=False, ax=ax1)



# Plot the PACF of savings on ax2

plot_pacf(savings, lags=10, zero=False, ax=ax2)



plt.show()
# Loop over p values from 0-3

for p in range(4):

  

  # Loop over q values from 0-3

    for q in range(4):

        try:

            # Create and fit ARMA(p,q) model

            model = SARIMAX(savings, order=(p,0,q), trend='c')

            results = model.fit()

        

            # Print p, q, AIC, BIC

            print(p, q, results.aic, results.bic)

        

        except:

            print(p, q, None, None)
# Create and fit model

model = SARIMAX(savings, order=(1,0,2), trend='c')

results = model.fit()



# Create the 4 diagostics plots

results.plot_diagnostics()

plt.show()



# Print summary

print(results.summary())
milk_production = pd.read_csv('/kaggle/input/week6dataset/milk_production.csv')

milk_production.head()
# Import seasonal decompose

from statsmodels.tsa.seasonal import seasonal_decompose



# Perform additive decomposition

decomp = seasonal_decompose(milk_production['pounds_per_cow'], 

                            freq=12)



# Plot decomposition

decomp.plot()

plt.show()
water = pd.DataFrame([24963, 27380, 32588, 25511, 32313, 31858, 23929, 30452, 28688, 24936, 34078, 25601, 27238, 28126, 32845, 26001, 29377, 33275, 28720, 31159, 23885, 29233, 39031, 25918, 27885, 28695, 30102, 30110, 32692, 28224, 30919, 30229, 28395, 32274, 32992, 26956, 28996, 28154, 27478, 32720, 30466, 27618, 29949, 29821, 28098, 32021, 29145, 29428, 28863, 28687, 33043, 29092, 28731, 34249, 31617, 30542, 30420, 32422, 33084, 29167, 30164, 32655, 34654, 29570, 29096, 33523, 30801, 35274, 31433, 32584, 32625, 27569, 36909, 30822, 36301, 32327, 33149, 33027, 29178, 35109, 29988, 38742, 38934, 28986, 35780, 31676, 35069, 33238, 36737, 32407, 33399, 37719, 30027, 37644, 39654, 28707, 28942, 31080, 26923, 32836, 32000, 30802, 31344, 32409, 33971, 33584, 30248, 30469, 30505, 31112, 31784, 31682, 30539, 36502, 32139, 30369, 36918, 32316, 33333, 30016, 30559, 31478, 35299, 30442, 32145, 35360, 34610, 33873, 37132, 32695, 34048, 32256, 27517, 33454, 38539, 31775], columns = ['water_consumers'])
# Create figure and subplot

fig, ax1 = plt.subplots()



# Plot the ACF on ax1

plot_acf(water['water_consumers'], lags = 25, zero=False,  ax=ax1)



# Show figure

plt.show()
# Subtract the rolling mean

water_2 = water - water.rolling(15).mean()



# Drop the NaN values

water_2 = water_2.dropna()



# Create figure and subplots

fig, ax1 = plt.subplots()



# Plot the ACF

plot_acf(water_2['water_consumers'], lags=25, zero=False, ax=ax1)



# Show figure

plt.show()
df1 = pd.DataFrame([-259.9, 651.9, 53.5, -418.1, 459.3, 2572.0, 897.8, -136.5, 793.8, -1.8, -464.6, 562.6, 2650.0, 807.0, -369.5, 894.8, -196.3, -615.0, 812.5, 3044.6, 762.3, -445.1, 984.9, -201.1, -879.6, 736.5, 3108.4, 533.9, -538.8, 1192.0, -349.8, -1202.7, 861.1, 2767.1, 766.8, -183.2, 1267.3, -502.6, -1355.3, 946.8, 3064.6, 780.5, 72.8, 1206.5, -370.5, -1378.2, 774.6, 3170.4, 1042.4, 279.2, 1376.2, -195.8, -1643.7, 941.6, 3158.9, 1250.8, 281.9, 1470.7, 46.9, -1425.1, 564.7, 2764.6, 1461.1, 227.3, 1779.1, 267.2, -1499.6, 692.5, 2291.8, 1406.8, -229.3, 1617.0, 481.4, -1513.9, 560.5, 1803.3, 1207.1, -341.3, 1355.5, 553.4, -1764.5, 259.4, 1606.9, 1278.8, -261.9, 1208.2, 851.6, -1609.9, 348.3, 1380.7], columns = ['Y'])
# Create a SARIMAX model

model = SARIMAX(df1, order=(1,0,0), seasonal_order=(1,1,0,7))



# Fit the model

results = model.fit()



# Print the results summary

print(results.summary())
df2 = pd.DataFrame([-1664.3, -1797.4, -2108.9, -2611.2, -3025.8, -2766.8, -2867.3, -3411.7, -3704.7, -3790.3, -4503.6, -4315.4, -5273.4, -5383.0, -5747.1, -6112.5, -6638.5, -6563.1, -6382.7, -6961.3, -6970.9, -7262.2, -6999.7, -7267.9, -6980.6, -7279.2, -7171.9, -7617.8, -7089.1, -7779.4, -7411.7, -8516.6, -8558.8, -8915.4, -8828.1, -8621.7, -8390.8, -8545.1, -8154.6, -8189.9, -8266.1, -8347.1, -8514.3, -8151.6, -8305.7, -7907.7, -7818.0, -7146.5, -6911.6, -6455.4, -6029.9, -6082.8, -5903.3, -5473.2, -5449.2, -5426.6, -5776.6, -5297.3, -5650.8, -5276.0, -4926.7, -4835.7, -4661.1, -4122.2, -3942.5, -3809.0, -3468.5, -3626.3, -3252.4, -3382.5, -3248.6, -3328.3, -3460.1, -3613.4, -3614.8, -3837.4, -3756.4, -3664.5, -3676.9, -4159.4], columns = ['Y'])
# Create a SARIMAX model

model = SARIMAX(df2, order = (2,1,1), seasonal_order = (1,0,0,4))



# Fit the model

results = model.fit()



# Print the results summary

print(results.summary())
df3 = pd.DataFrame([2141.5, 1557.6, 2715.1, 1313.5, 777.6, 2476.4, 2640.6, 2415.6, 1843.9, 1946.8, 2190.8, 2422.0, 5287.3, 4873.0, 6134.6, 4659.4, 3984.0, 5699.3, 6003.3, 5810.2, 5422.5, 5585.2, 5849.1, 6151.7, 9203.4, 8623.9, 9859.8, 8293.2, 7491.6, 9013.1, 9285.1, 9148.3, 8582.1, 8788.6, 9045.2, 9375.2, 12368.8, 11716.1, 12989.1, 11395.0, 10676.0, 12216.7, 12579.4, 12396.4, 12041.2, 12356.7, 12554.1, 12901.8, 15781.6, 15106.7, 16339.1, 14705.4, 14039.6, 15448.2, 15628.4, 15342.5, 14863.2, 15165.4, 15369.0, 15733.9, 18507.2, 17757.6, 18980.3, 17364.9, 16977.9, 18468.0, 18756.5, 18606.6, 18228.3, 18481.9, 18543.6, 18938.8, 21818.7, 21081.8, 22313.7, 20751.2, 20316.2, 21761.6, 21861.8, 21511.0, 21102.3, 21404.7, 21528.0, 22004.0, 25033.4, 24455.4, 25831.4, 24554.4, 24371.8, 25742.6, 25772.6, 25365.1, 24976.2, 25299.7, 25521.3, 26135.7, 29183.6, 28621.2, 30025.4, 28766.1], columns = ['Y'])
# Create a SARIMAX model

model = SARIMAX(df3, order = (1,1,0), seasonal_order = (0,1,1,12))



# Fit the model

results = model.fit()



# Print the results summary

print(results.summary())
aus_employment = pd.DataFrame([5985.7, 6040.6, 6054.2, 6038.3, 6031.3, 6036.1, 6005.4, 6024.3, 6045.9, 6033.8, 6125.4, 5971.3, 6050.7, 6096.2, 6087.7, 6075.6, 6095.7, 6103.9, 6078.5, 6157.8, 6164.0, 6188.8, 6257.2, 6112.9, 6207.2, 6278.7, 6224.9, 6273.4, 6269.9, 6314.1, 6281.4, 6360.0, 6320.2, 6342.0, 6426.6, 6253.0, 6356.5, 6428.1, 6426.3, 6412.4, 6413.9, 6425.3, 6393.7, 6502.7, 6445.3, 6433.3, 6506.9, 6355.5, 6432.4, 6497.4, 6431.6, 6440.9, 6414.3, 6425.9, 6379.3, 6443.5, 6421.1, 6367.0, 6370.2, 6172.2, 6264.1, 6310.4, 6254.5, 6272.8, 6266.5, 6295.0, 6241.1, 6358.2, 6336.2, 6377.5, 6456.4, 6251.4, 6365.4, 6503.2, 6477.6, 6489.7, 6499.0, 6528.7, 6466.1, 6579.8, 6553.2, 6576.1, 6636.0, 6452.4, 6595.7, 6657.4, 6588.8, 6658.0, 6659.4, 6703.4, 6675.6, 6814.7, 6771.1, 6882.0, 6910.8, 6753.6, 6861.9, 6961.9, 6997.9, 6979.0, 7007.7, 6991.5, 6918.6, 7040.6, 7030.4, 7034.2, 7116.8, 6902.5, 7022.3, 7133.4, 7109.6, 7103.6, 7128.9, 7175.6, 7092.3, 7186.5, 7177.4, 7182.2, 7330.7, 7169.4, 7247.4, 7397.4, 7383.4, 7354.9, 7378.3, 7383.1, 7353.4, 7503.2, 7477.3, 7508.7, 7622.9, 7424.9, 7569.7, 7638.3, 7683.2, 7729.3, 7720.5, 7751.8, 7727.6, 7854.4, 7817.8, 7870.7, 7941.6, 7712.6, 7809.1, 7877.5, 7894.3, 7916.1, 7910.0, 7932.9, 7825.0, 7925.5, 7870.5, 7849.9, 7941.2, 7668.8, 7739.3, 7746.5, 7750.5], columns = ['people_employed'])
# Take the first and seasonal differences and drop NaNs

aus_employment_diff = aus_employment.diff().diff(12).dropna()
# Create the figure 

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))



# Plot the ACF on ax1

plot_acf(aus_employment_diff, lags = 11, zero = False, ax = ax1)



# Plot the PACF on ax2

plot_pacf(aus_employment_diff, lags = 11, zero = False, ax = ax2)



plt.show()
# Make list of lags

lags = [12, 24, 36, 48, 60]



# Create the figure 

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))



# Plot the ACF on ax1

plot_acf(aus_employment_diff, lags=lags, ax=ax1)



# Plot the PACF on ax2

plot_pacf(aus_employment_diff, lags=lags, ax=ax2)



plt.show()
import numpy
dates = [numpy.datetime64('1973-10-01T00:00:00.000000000'), numpy.datetime64('1973-11-01T00:00:00.000000000'), numpy.datetime64('1973-12-01T00:00:00.000000000'), numpy.datetime64('1974-01-01T00:00:00.000000000'), numpy.datetime64('1974-02-01T00:00:00.000000000'), numpy.datetime64('1974-03-01T00:00:00.000000000'), numpy.datetime64('1974-04-01T00:00:00.000000000'), numpy.datetime64('1974-05-01T00:00:00.000000000'), numpy.datetime64('1974-06-01T00:00:00.000000000'), numpy.datetime64('1974-07-01T00:00:00.000000000'), numpy.datetime64('1974-08-01T00:00:00.000000000'), numpy.datetime64('1974-09-01T00:00:00.000000000'), numpy.datetime64('1974-10-01T00:00:00.000000000'), numpy.datetime64('1974-11-01T00:00:00.000000000'), numpy.datetime64('1974-12-01T00:00:00.000000000'), numpy.datetime64('1975-01-01T00:00:00.000000000'), numpy.datetime64('1975-02-01T00:00:00.000000000'), numpy.datetime64('1975-03-01T00:00:00.000000000'), numpy.datetime64('1975-04-01T00:00:00.000000000'), numpy.datetime64('1975-05-01T00:00:00.000000000'), numpy.datetime64('1975-06-01T00:00:00.000000000'), numpy.datetime64('1975-07-01T00:00:00.000000000'), numpy.datetime64('1975-08-01T00:00:00.000000000'), numpy.datetime64('1975-09-01T00:00:00.000000000'), numpy.datetime64('1975-10-01T00:00:00.000000000')]
wisconsin_test = pd.DataFrame([374.5, 380.2, 384.6, 360.6, 354.4, 357.4, 367.0, 375.7, 381.0, 381.2, 383.0, 384.3, 387.0, 391.7, 396.0, 374.0, 370.4, 373.2, 381.1, 389.9, 394.6, 394.0, 397.0, 397.2, 399.4], columns = ['number_in_employment'])
wisconsin_test.index = dates
# Create a SARIMAX model

model = SARIMAX(wisconsin_test['number_in_employment'], order = (3, 1, 2))



# Fit the model

results = model.fit()



# Create ARIMA mean forecast

arima_mean = results.predict(0)



plt.plot(dates, arima_mean, label='ARIMA')

plt.plot(wisconsin_test, label='observed')

plt.legend()

plt.show()
# Import pmdarima as pm

import pmdarima as pm

import warnings

warnings.filterwarnings('ignore')
# Create auto_arima model

model1 = pm.auto_arima(df1,

                      seasonal=True, m=7,

                      d=0, D=1, 

                 	  max_p=2, max_q=2,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True)

                       

# Print model summary

print(model1.summary())
# Create model

model2 = pm.auto_arima(df2,

                      seasonal=False, 

                      d=1, 

                      trend='c',

                 	  max_p=2, max_q=2,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True) 



# Print model summary

print(model2.summary())
# Create model for SARIMAX(p,0,q)(P,1,Q)7

model3 = pm.auto_arima(df3,

                      seasonal=True, m=7,

                      d=1, D=1, 

                      start_p=1, start_q=1,

                      max_p=1, max_q=1,

                      max_P=1, max_Q=1,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True) 



# Print model summary

print(model3.summary())
# Create a SARIMAX model

model = SARIMAX(candy_train['IPG3113N'], order = (1, 0, 1), seasonal_order = (0, 1, 1, 12))



# Fit the model

results = model.fit()



# Print the results summary

print(results.summary())
# Import joblib

import joblib



# Set model name

filename = "candy_model.pkl"



# Pickle it

joblib.dump(model, filename)
# Import

import joblib



# Set model name

filename = "candy_model.pkl"



# Load the model back in

loaded_model = joblib.load(filename)
# Update the model

loaded_model.update(candy_test['IPG3113N'])
co2 = pd.read_csv('/kaggle/input/week6dataset/co2.csv', parse_dates = ['date'])

co2 = co2.set_index('date')

co2.head()
# Import model class

from statsmodels.tsa.statespace.sarimax import SARIMAX



# Create model object

model = SARIMAX(co2, 

                order=(1,1,1), 

                seasonal_order=(0,1,1,12), 

                trend='c')

# Fit model

results = model.fit()
# Plot common diagnostics

results.plot_diagnostics()

plt.show()
# Create forecast object

predicted_mean = results.predict(40)



# Extract the forecast dates

dates = predicted_mean.index
plt.figure()



# Plot past CO2 levels

plt.plot(co2.index.values, co2.values, label='past')



# Plot the prediction means as line

plt.plot(dates, predicted_mean, label='predicted')



# Plot legend and show figure

plt.legend()

plt.show()