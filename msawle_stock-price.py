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
# Raw data was exported from YAHOO Finance
# https://in.finance.yahoo.com/quote/TATAMOTORS.BO/history/

data = pd.read_csv('/kaggle/input/TATAMOTORS.BO.csv', index_col = 'Date', parse_dates = ['Date'])
data.head()
data.describe(include = 'all')
data.isna().sum()
print(np.where(data['Open'].isna()))
print(np.where(data['High'].isna()))
print(np.where(data['Low'].isna()))
print(np.where(data['Close'].isna()))
print(np.where(data['Adj Close'].isna()))
print(np.where(data['Volume'].isna()))
data.dropna(how = 'any', inplace = True)
data
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight') 
%matplotlib inline

plt.figure(figsize = (20,8))

data['Open'].plot()
data['High'].plot()
data['Low'].plot()
# data['Volume'].plot()


plt.title("5 Year Price Movement")
plt.legend()
plt.show()
data_monthly = data.resample('M').mean()
data_weekly = data.resample('W').mean()
data_quarterly = data.resample('Q').mean()
# Monthly price movement over 5 Years

plt.figure(figsize = (20,8))
plt.plot(data_monthly['Open'], label = 'Open')
plt.plot(data_monthly['Close'], label = 'Close')
plt.plot(data_monthly['High'], label = 'High')
plt.legend()
plt.title('Monthly price movement over 5 Years')
plt.xlabel('Year')
plt.ylabel('Stock Price in INR')
plt.show()
# Weekly price movement over 5 Years

plt.figure(figsize = (20,8))
plt.plot(data_weekly['Open'], label = 'Open')
plt.plot(data_weekly['Close'], label = 'Close')
plt.plot(data_weekly['High'], label = 'High')
plt.legend()
plt.title('Weekly price movement over 5 Years')
plt.xlabel('Year')
plt.ylabel('Stock Price in INR')
plt.show()
data_YTD = data['2020']
data.plot(subplots = True, figsize = (20,16))
plt.title('Tata Motors stock attributes from 2016 to 2020')
plt.savefig('stocks.png')
plt.show()
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

import statsmodels.api as sm

plt.figure(figsize = (20,8))
y = data['Open'].resample('W').mean()

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')
decomposition.plot()
plt.show()

y1 = data['Open'].resample('7D').mean()
y1 = y1.ffill()

rcParams['figure.figsize'] = 18, 8
plt.figure(figsize = (20,8))


decomposition_1 = sm.tsa.seasonal_decompose(y1, model = 'additive')
decomposition_1.plot()
plt.show()


# Rolling average stock price
plt.figure(figsize = (20,8))
data['close_7'] = data['Close'].rolling(10).mean()
data['close_30'] = data['Close'].rolling(30).mean()

plt.plot(data['Close'], label = 'Daily closing Price')
plt.plot(data['close_7'], label = '7 Days rolling mean')
plt.plot(data['close_30'], label = '30 Days rolling mean')

plt.legend()
plt.show()
# Calculating the daily return
plt.figure(figsize = (20,8))
data['Daily_Return'] = data['Close'].pct_change() * 100
plt.plot(data['Daily_Return'], label = "Daily %age return", linestyle = '--', marker = 'o')
plt.xlabel('Date')
plt.ylabel('%age return')
plt.legend()
plt.show()

# Calculating the daily return
plt.figure(figsize = (20,8))
plt.plot(data.loc['2019':, 'Daily_Return'], label = "Daily %age return", linestyle = '--', marker = 'o')
plt.xlabel('Date')
plt.ylabel('%age return')
plt.legend()
plt.show()
# Filtering the instances / dates when the daily return exceeded 10 % in a Single day

data[data['Daily_Return'] >= 10]
# Filtering the instances / dates when the daily return tanked by 10 % in a Single day

data[data['Daily_Return'] <= -10]
sns.distplot(data['Daily_Return'].dropna(), bins =50, color = 'purple')
print(data['Daily_Return'].quantile(.05))
print(data['Daily_Return'].quantile(.99))
print(data['Daily_Return'].quantile(.95))
plt.style.use(['ggplot'])

# to suppress warnings
import warnings
warnings.filterwarnings('ignore')

# sets the plot size to 12x8
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,8)

# plots the ACF for the differenced data over the entire train period
pd.plotting.autocorrelation_plot(data.Close.diff().dropna(), linewidth=1.0)
plt.ylim([-0.25, 0.25])
import fbprophet
data_close = pd.DataFrame(data['Close'])
data_close.reset_index(inplace = True)
data_close.rename(columns = {'Date': 'ds', 'Close': 'y'}, inplace = True)
data_close.head()

# We created a sample time series dataframe for the clsoing price from the earlier dataframe that included complete details of the stock price
# for fbprophet, it is mandatory to name Date column as 'ds' and the dependent variable as 'y'
model = fbprophet.Prophet(changepoint_prior_scale = 0.05)
model.fit(data_close)
data_forecast = model.make_future_dataframe(periods = 100, freq = 'D')
data_forecast = model.predict(data_forecast)
plt.style.use('ggplot')
model.plot(data_forecast, xlabel = 'Date', ylabel = 'Closing Price')
plt.legend()
plt.title('Daily Closing Price Prediction')
data_forecast[data_forecast['ds'] == '2020-06-10'].loc[:,['ds', 'yhat']]