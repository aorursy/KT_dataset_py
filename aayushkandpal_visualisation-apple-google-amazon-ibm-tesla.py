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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
# Importing our Dataset
stocks_df = pd.read_csv('../input/applegoogleamazonibmteslasp500-2012-2020/stock.csv')
stocks_df
stocks_df.columns
# A custom function for making interactive plots
def interactive_plot(df, title):
  fig = px.line(title = title)
  
  
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i) 

  fig.show()
interactive_plot(stocks_df, 'Stock_Prices')
# This function will help us visulaize daily changes in the stock prices by subtracting the stock price from its previous day price
# and dividing it by the prev value to get the net change. Multiply by 100.
def daily_return(df):
  df_daily_return = df.copy()

  
  for i in df.columns[1:]:
    
    
    for j in range(1, len(df)):

      
      df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    
    
    df_daily_return[i][0] = 0
  
  return df_daily_return
stocks_daily_return = daily_return(stocks_df)
stocks_daily_return
# Zoom in to see more clearly. In fact to some extent you can also understand how volatile a given stock is by looking
#  at its deviation from the mean
interactive_plot(stocks_daily_return, 'Daily Returns on Stocks')
# Distribution of stock prices
stocks_df.hist(figsize=(15, 15), bins = 50,color='orange');
stocks_df.skew()
stocks_daily_return.skew()
# this is the volalitity of every stock 
stocks_daily_return.hist(figsize=(15,15),color='b',bins=50)
stock = stocks_daily_return.copy()


stock = stock.drop(columns = ['Date'])

data = []


for i in stock.columns:
  data.append(stocks_daily_return[i].values)
data
# visualzing the volatility of all stocks in a single graph
fig = ff.create_distplot(data, stock.columns)
fig.show()
# By what factor did the value of the stock grown since 2012
def normalize(df):
  x = df.copy()

  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

interactive_plot(normalize(stocks_df), 'X Returns')
# A custom plot to visualise all the different indicators in one graph, you could use many more indicators but that would be tough to interpret.
plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['AAPL'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['AAPL'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['AAPL'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['AAPL'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['AAPL'])
stocks_df['ma20'] = stocks_df['AAPL'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['AAPL'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()
plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['AMZN'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['AMZN'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['AMZN'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['AMZN'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['AMZN'])
stocks_df['ma20'] = stocks_df['AMZN'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['AMZN'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()
plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['GOOG'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['GOOG'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['GOOG'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['GOOG'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['GOOG'])
stocks_df['ma20'] = stocks_df['GOOG'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['GOOG'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()

plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['TSLA'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['TSLA'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['TSLA'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['TSLA'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['TSLA'])
stocks_df['ma20'] = stocks_df['TSLA'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['TSLA'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()

plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['IBM'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['IBM'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['IBM'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['IBM'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['IBM'])
stocks_df['ma20'] = stocks_df['IBM'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['IBM'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()

plt.figure(figsize=(20,8))
stocks_df['ma50'] = stocks_df['sp500'].rolling(window=50).mean()
stocks_df['ma100'] = stocks_df['sp500'].rolling(window=100).mean()
stocks_df['ma200'] = stocks_df['sp500'].rolling(window=200).mean()
stocks_df['ma300'] = stocks_df['sp500'].rolling(window=300).mean()

plt.plot(stocks_df['ma50'],label='MA 50', color='r',linestyle='--')
plt.plot(stocks_df['ma100'],label='MA 100', color='g',linestyle='--')
plt.plot(stocks_df['ma200'],label='MA 200', color='y',linestyle='--')
plt.plot(stocks_df['ma300'],label='MA 300', color='black',linestyle='--')
plt.plot(stocks_df['sp500'])
stocks_df['ma20'] = stocks_df['sp500'].rolling(window=20).mean()
stocks_df['20sd'] = stocks_df['sp500'].rolling(window=20).std()
stocks_df['upper_band'] = stocks_df['ma20'] + (stocks_df['20sd']*2)
stocks_df['lower_band'] = stocks_df['ma20'] - (stocks_df['20sd']*2)
plt.plot(stocks_df['lower_band'],label='Lower_Band',color='black',linestyle='-')
plt.plot(stocks_df['upper_band'],label='Upper_Band',color='black',linestyle='-')
plt.legend()
plt.xlabel('Duration 12th Jan 2012 to 11th Aug 2020, Number of days -------')
plt.ylabel('Stock_Price')
plt.show()

plt.figure(figsize=(10, 10))
matrix=stocks_df.corr()
sns.heatmap(matrix,annot=True)


plt.figure(figsize=(10, 10))
matrix2=stocks_daily_return.corr()
sns.heatmap(matrix2,annot=True)
