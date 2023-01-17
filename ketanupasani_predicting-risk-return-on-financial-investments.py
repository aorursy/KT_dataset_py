# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Settings to produce nice plots in a Jupyter notebook

plt.style.use('fivethirtyeight')

%matplotlib inline



# Reading in the data

stock_data = pd.read_csv('../input/sharpe-ratio-dataset/stock_data.csv',

                          parse_dates=['Date'], 

                          index_col=['Date']).dropna()



benchmark_data = pd.read_csv('../input/sharpe-ratio-dataset/benchmark_data.csv', 

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

stock_data.plot(subplots=True)

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
result = pd.merge(stock_data,benchmark_data, on = 'Date',how = 'left')

result.info()
result = result.reset_index()
result.info()
result['Date'].min()
result['Date'].max()
#Merging both data frames.

result = pd.merge(stock_data,benchmark_data, on = 'Date',how = 'left')

result
result.info()

result = result.reset_index()
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
from sklearn.linear_model import LinearRegression

model_fb = LinearRegression()

model_fb.fit(result[['S&P 500']],result['Facebook'])

#Sklearn wants x value to be in 2D.
plt.scatter(result['S&P 500'],result['Facebook'])

plt.plot(result[['S&P 500']],model_fb.predict(result[['S&P 500']]),c='r')

plt.xlabel('S&P 500')

plt.ylabel('Facebook')

plt.show()
from sklearn.linear_model import LinearRegression

model_amazon = LinearRegression()

model_amazon.fit(result[['S&P 500']],result['Amazon'])

#Sklearn wants x value to be in 2D.
plt.scatter(result['S&P 500'],result['Amazon'])

plt.plot(result[['S&P 500']],model_amazon.predict(result[['S&P 500']]),c='r')

plt.xlabel('S&P 500')

plt.ylabel('Amazon')

plt.show()
maxValuesobj = stock_data.max()

minValuesObj = stock_data.min()



maxValuesobj-minValuesObj
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
df_fb = result[['Date','Facebook']]

df_fb
df_fb = df_fb.rename(columns={'Date':'ds','Facebook':'y'})

df_fb
import fbprophet

op = fbprophet.Prophet(changepoint_prior_scale=0.50)

op.fit(df_fb)
forecast = op.make_future_dataframe(periods = 30,freq='D')

forecast = op.predict(forecast)
op.plot(forecast,xlabel='Date',ylabel='Facebook')

plt.title('Facebook Stock Prediction');