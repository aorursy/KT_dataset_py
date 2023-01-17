import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.stattools import adfuller, kpss
# numple of data point to be generated

N = 100000
# AR and MA coefficients

ar_coef = [1, 0.5]

ma_coef = [1]

arma_process = ArmaProcess(ar_coef, ma_coef)
# generate stationary data

stationary_data = arma_process.generate_sample(N)
# plot the whole data and first 100 data points

plt.figure(figsize=(20,5))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

ax1.plot(stationary_data)

ax1.set_title('Plot of all data points')

ax2.plot(stationary_data[:100])

ax2.set_title('Plot of first 100 data points')
mean_list = []

var_list = []



sample_size = 100



for i in range(1, N-sample_size):

    start = i

    end = i+sample_size

    sample = stationary_data[start:end]

    mean_list.append(sample.mean())

    var_list.append(sample.var())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))



sns.kdeplot(mean_list, ax=ax1, color='green')

ax1.axvline(np.mean(stationary_data), color='red', linestyle='--')

ax1.set_title('Sample mean distribution')



# the dashed lines represents the true mean and variance in the 2 figures



sns.kdeplot(var_list, ax=ax2)

ax2.axvline(np.var(stationary_data), color='red', linestyle='--')

ax2.set_title('Sample varriance distribution')
print('True mean: {}'.format(np.mean(stationary_data)))

print('True variance: {}'.format(np.var(stationary_data)))
# test if time series is stationary

def check_stationary(series):

    '''

        Perform augmented ickey fuller test. The p-value (2nd element of the returned result) 

        close to 0 (<0.05 - typical significant level) denotes a stationary time series

    '''

    res = adfuller(series)

    if(res[1] < 0.05):

        print('The series is stationary with p-value={}'.format(res[1]))

    else:

        print('The series is non-stationary with p-value={}'.format(res[1]))
# dickey fuller test on the series

check_stationary(stationary_data)
# create a line with slope = 7 and intercept = 5

linear_trend = 150*np.linspace(0, 1, N) + 10
# add the trend to the stationary data to make a new series with trend

trend_data = linear_trend + stationary_data
n_samples = 5000



trend_data = linear_trend + stationary_data



fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20, 10))



ax1.plot(linear_trend[:n_samples])

ax1.set_title('Trend pattern'.format(n_samples))



ax2.plot(trend_data[:n_samples])

ax2.set_title('First {} points of a trend time series'.format(n_samples))

# Test if the series is stationary

check_stationary(trend_data)
mean_list = []



sample_size = 100



for i in range(1, N-sample_size):

    start = i

    end = i+sample_size

    sample = trend_data[start:end]

    mean_list.append(sample.mean())
plt.plot(mean_list)

plt.title('Sample means over time of a trend time series')
T = 365 # a time length for 1 cycle

n_periods = N // T + 1
sine_seasonality = np.sin(2*np.pi*np.linspace(0, 1*n_periods, T*n_periods))

# trim to get size that equals to N

sine_seasonality = sine_seasonality[:N]

# add an upward trend

sine_seasonality += np.arange(0, N)*0.001
n_samples = 5000



seasonal_data = sine_seasonality + stationary_data

seasonal_samples = seasonal_data[:n_samples]



seasonal_data = sine_seasonality + stationary_data



fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20, 10))



ax1.plot(sine_seasonality[:n_samples])

ax1.set_title('First {} points of seasonal pattern'.format(n_samples))



ax2.plot(seasonal_samples)

for i in range(n_periods):

    if(i > 5000//T):

        break

    ax2.axvline(i*T, color='red', linestyle='--')

ax2.set_title('First {} points of a seasonal time series'.format(n_samples))

check_stationary(seasonal_data)