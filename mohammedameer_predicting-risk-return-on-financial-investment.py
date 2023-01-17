# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
%matplotlib inline

# Reading in the data
stock_data = pd.read_csv('../input/stock_data.csv',
                          parse_dates=['Date'], 
                          index_col=['Date']).dropna()

benchmark_data = pd.read_csv('../input/benchmark_data.csv', 
                             parse_dates=['Date'], 
                             index_col=['Date']).dropna()
# Display summary for stock_data
print('Stocks\n')
stock_data.info()
stock_data.head()


# Display summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
benchmark_data.head()

# visualize the stock_data
stock_data.plot()  # Pandas Plotting
plt_title = 'Stock Data'
plt.show()

# summarize the stock_data
stock_data.describe()
# plot the benchmark_data
benchmark_data.plot()
plt_title = 'S&P 500'
plt.show()


# summarize the benchmark_data
benchmark_data.describe()

stock_data.pct_change()
# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns
stock_returns.plot()
plt.show()


# summarize the daily returns
stock_returns.describe()
# calculate daily benchmark_data returns
# ... YOUR CODE FOR TASK 6 HERE ...
sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns
sp_returns.plot()
plt.show()


# summarize the daily returns
sp_returns.describe()

stock_returns.sub(sp_returns, axis=0)
# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plot the excess_returns
excess_returns.plot()
plt.show()


# summarize the excess_returns
excess_returns.describe()

# calculate the mean of excess_returns 
# ... YOUR CODE FOR TASK 8 HERE ...
avg_excess_return = excess_returns.mean()

# plot avg_excess_returns
avg_excess_return.plot.bar()
plt_title = 'Mean of the Return Difference'
plt.show()

# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations
sd_excess_return.plot()
plt_title = 'Standard Deviation of the Return Difference'
plt.show()
# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot()
plt_title = 'Annualized Sharpe Ratio: Stocks vs S&P 500'
plt.show()
# Merging both the dataframes
result = pd.merge(stock_data,benchmark_data,on ='Date',how='left')
result
result.info()
result = result.reset_index()
result
result.info()
plt.figure(figsize=(8,4))
plt.plot(result['Date'],result['Amazon'],label='Amazon')
plt.plot(result['Date'],result['Facebook'],label='Facebook')
plt.plot(result['Date'],result['S&P 500'],label='S&P 500')
plt.legend()
plt.show()
plt.scatter(result['S&P 500'],result['Facebook'])
plt.xlabel('S&P 500')
plt.ylabel('Facebook')
plt.show()
plt.scatter(result['S&P 500'],result['Amazon'])
plt.xlabel('S&P 500')
plt.ylabel('Amazon')
plt.show()
# Linear Regression for S&P 500 vs Facebook -  Best Fit Line
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(result[['S&P 500']],result['Facebook']) # Sklearn wants x value to be in 2D
plt.scatter(result['S&P 500'],result['Facebook'])
plt.plot(result[['S&P 500']],model.predict(result[['S&P 500']]),c='r')
plt.xlabel('S&P 500')
plt.ylabel('Facebook')
plt.show()
# MSE,RSME,MAE
result['Date'].max()
result['Date'].min()
# Date when there was Max Value of Amazon
result[result['Amazon']==result['Amazon'].max()]
# Date when there was Min Value of Amazon
result[result['Amazon']==result['Amazon'].min()]
# Date when there was Min Value of Facebook
result[result['Facebook']==result['Facebook'].min()]
# Forecasting
result
df_am = result[['Date','Amazon']]
df_am
df_am = df_am.rename(columns={'Date':'ds','Amazon':'y'})
df_am
import fbprophet
op = fbprophet.Prophet(changepoint_prior_scale=0.50)
op.fit(df_am)
forecast = op.make_future_dataframe(periods = 30,freq='D')
forecast = op.predict(forecast)
op.plot(forecast,xlabel='Date',ylabel='Amazon')
plt.title('Amazon Stock Prediction');
