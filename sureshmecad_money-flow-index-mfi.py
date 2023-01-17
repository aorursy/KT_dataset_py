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
# import the libraries

import pandas as pd

import numpy as np

from pandas import datetime



import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')



import warnings

warnings.filterwarnings('ignore')
# Get the data



df = pd.read_csv("/kaggle/input/apple-aapl-historical-stock-data/HistoricalQuotes.csv", index_col='Date', parse_dates=True)

df.head()
df.shape
df.info()
# Cleaning data



df = df.rename(columns={' Close/Last':'Close', ' Volume':'Volume', ' Open': 'Open', ' High':'High', ' Low':'Low'})

df['Close'] = df['Close'].str.replace('$', '').astype('float')

df['Open'] = df['Open'].str.replace('$', '').astype('float')

df['High'] = df['High'].str.replace('$', '').astype('float')

df['Low'] = df['Low'].str.replace('$', '').astype('float')

df.head()
# Let's go ahead and use sebron for a quick correlation plot for the daily returns

sns.heatmap(df.corr(), annot=True, cmap='summer')
sns.pairplot(df, kind='reg')
return_fig = sns.PairGrid(df.dropna())

return_fig.map_upper(plt.scatter, color='purple')

return_fig.map_lower(sns.kdeplot, cmap='cool_d')

return_fig.map_diag(plt.hist, bins=30)
# Visually show the stock price

plt.figure(figsize=(15,10))

plt.plot(df['Close'], label='Close Price', alpha=1)

plt.title('APPL Close Price History')

plt.xlabel('Date')

plt.ylabel('Close Price')

plt.legend(loc='upper left')

plt.xticks(rotation=45)

plt.show()
# Create the simple moving average with a 30 days window

SMA30 = pd.DataFrame()

SMA30['Close'] = df['Close'].rolling(window=30).mean()

SMA30
# Create the simple moving average with a 100 days window

SMA100 = pd.DataFrame()

SMA100['Close'] = df['Close'].rolling(window=100).mean()

SMA100
# Visually show the stock price

plt.figure(figsize=(15,10))

plt.plot(df['Close'], label='Close Price')

plt.plot(SMA30['Close'], label='SMA30')

plt.plot(SMA100['Close'], label='SMA100')

plt.title('Close Price History')

plt.xlabel('Timestamp')

plt.ylabel('Price')

plt.xticks(rotation=45)

plt.legend(loc='upper left')

plt.show()
# Calculate typical price

typical_price = (df['Close'] + df['High'] + df['Low']) / 3

typical_price
# Get the period

period = 14



# Calculate the money flow

money_flow = typical_price * df['Volume']

money_flow
# Get all of the positive and negative money flows



positive_flow = []

negative_flow = []



# Loop through the typical price

for i in range(1, len(typical_price)):

    if typical_price[i] > typical_price[i-1]:

        positive_flow.append(money_flow[i-1])

        negative_flow.append(0)

        

    elif typical_price[i] < typical_price[i-1]:

        negative_flow.append(money_flow[i-1])

        positive_flow.append(0)

        

    else:

        positive_flow.append(0)

        negative_flow.append(0)
# Get all of the positive and negative money flows within the time period



positive_mf = []

negative_mf = []



for i in range(period-1, len(positive_flow)):

    positive_mf.append( sum(positive_flow[i + 1- period : i+1]))

    

for i in range(period-1, len(negative_flow)):

    negative_mf.append( sum(negative_flow[i + 1- period : i+1]))
# Calculate the money flow index

MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf) ))

MFI
# Visually show the MFI



df2 = pd.DataFrame()

df2['MFI'] = MFI



# Create the plot



plt.figure(figsize=(15,12))

plt.plot(df2['MFI'], label='MFI')

plt.axhline(10, linestyle= '--', color='orange')

plt.axhline(20, linestyle= '--', color='blue')

plt.axhline(80, linestyle= '--', color='blue')

plt.axhline(90, linestyle= '--', color='orange')

plt.title('MFI', fontsize=18)

plt.ylabel('MFI Values')

plt.show()
# Create a new Dataframe



new_df = pd.DataFrame()

new_df = df[period:]

new_df['MFI'] = MFI



# Show the new dataframe



new_df
# Create a function to get the buy and sell signals

def get_signal(data, high, low):

    buy_signal = []

    sell_signal = []

          

    for i in range(len(data['MFI'])):

        if data['MFI'][i] > high:

            buy_signal.append(np.nan)

            sell_signal.append(data['Close'][i])

          

        elif data['MFI'][i] < low:

       

            buy_signal.append(data['Close'][i])

            sell_signal.append(np.nan)

          

        else:

            sell_signal.append(np.nan)

            buy_signal.append(np.nan)

      



    return (buy_signal, sell_signal)
# Add new columns (Buy and Sell)



new_df['Buy'] = get_signal(new_df, 80, 20)[0]

new_df['Sell'] = get_signal(new_df, 80, 20)[1]



# Show the data



new_df
# Plot the data



plt.figure(figsize=(15,12))

plt.plot(new_df['Close'], label='Close Price', alpha=0.5)

plt.scatter(new_df.index, new_df['Buy'], label='Buy Signal', color='green', marker='^', alpha=1)

plt.scatter(new_df.index, new_df['Sell'], label='Sell Signal', color='red', marker='^', alpha=1)

plt.title('Apple Close Price', fontsize=18)

plt.xticks(rotation=45)

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price', fontsize=18)

plt.legend(loc='upper left')

plt.show()



# Create the plot



plt.figure(figsize=(15,12))

plt.plot(new_df['MFI'], label='MFI')

plt.axhline(10, linestyle= '--', color='orange')

plt.axhline(20, linestyle= '--', color='blue')

plt.axhline(80, linestyle= '--', color='blue')

plt.axhline(90, linestyle= '--', color='orange')

plt.title('MFI', fontsize=18)

plt.ylabel('MFI Values')

plt.show()