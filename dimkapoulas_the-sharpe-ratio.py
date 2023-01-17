# Importing required modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Settings to produce nice plots in a Jupyter notebook

plt.style.use('fivethirtyeight')

%matplotlib inline



# Reading in the data

stock_data = pd.read_csv('../input/stock_data.csv', parse_dates=['Date'], index_col='Date')

benchmark_data = pd.read_csv('../input/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna()
# Display summary for stock_data

print('Stocks\n')

print(stock_data.info())

# Display summary for benchmark_data

print('\nBenchmarks\n')

print(benchmark_data.info())
# visualize the stock_data

stock_data.plot(subplots=True, title='Stock Data')



# summarize the stock_data

print(stock_data.describe())

plt.show()

# plot the benchmark_data

benchmark_data.plot(title='S&P 500')



# summarize the benchmark_data

print(benchmark_data.describe())

plt.show()
# calculate daily stock_data returns

stock_returns = stock_data.pct_change()



# plot the daily returns

stock_returns.plot()

plt.show()

# summarize the daily returns

stock_returns.describe()
# calculate daily benchmark_data returns

sp_returns = benchmark_data['S&P 500'].pct_change()



# plot the daily returns

sp_returns.plot(c='green')

plt.show()

# summarize the daily returns

# ... YOUR CODE FOR TASK 6 HERE ...

# calculate the difference in daily returns

excess_returns = stock_returns.sub(sp_returns, axis=0)



# plot the excess_returns

excess_returns.plot()

plt.show()

# summarize the excess_returns

excess_returns.describe()
# calculate the mean of excess_returns 

avg_excess_return = excess_returns.mean(axis=0, skipna=True)



# plot avg_excess_returns

avg_excess_return.plot.bar(title='Return Difference')

plt.show()
# calculate the standard deviations

sd_excess_return = excess_returns.std(axis=0, skipna=True)



# plot the standard deviations

sd_excess_return.plot.bar(title='Standard Deviation of the Return')

plt.show()
# calculate the daily sharpe ratio

daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)



# annualize the sharpe ratio

annual_factor = np.sqrt(252)

annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)



# plot the annualized sharpe ratio

annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ration: Stocks vs S&P 500')

plt.show()