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
amazon = pd.read_csv('../input/historical-amazon-stock-prices/AMZN.csv')

amazon.head()
amazon.info()

print(amazon.shape)
amazon.isnull().sum()
amazon.describe()
amazon['Date'] = pd.to_datetime(amazon['Date'])

amazon = amazon.set_index('Date')

amazon['Year'] = amazon.index.year

amazon['Month'] = amazon.index.month

amazon['Day'] = amazon.index.day

amazon.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
print(plt.style.available)
plt.style.use('seaborn-dark')
amazon[['Open', 'High', 'Low', 'Close']].rolling(window = 1).mean().plot(title = 'Open, High, Low, Close Timeline', subplots=True, figsize=(15,12))

plt.show()
s = amazon['2020'][['Open', 'High', 'Low', 'Close']].rolling(window = 1).mean().plot(title = 'Open, High, Low, Close Comparison in 2020', subplots=True, figsize=(15,12))

plt.show()
amazon= amazon.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year', 'Month', 'Day'], axis=1)
f, (ax1,ax2) = plt.subplots(2, figsize=(15,12))

days = range(252)



ax1.plot(days, amazon['1998']['Close'])

ax1.set(title = 'Stocks in 1998', xlabel='Days', ylabel='Stock Value')



ax2.plot(days, amazon['2019']['Close'])

ax2.set(title = 'Stocks in 2019', xlabel='Days', ylabel='Stock Value')



plt.show()
fig = plt.figure(figsize=(12,10))



days = range(184)



plt.plot(days, amazon['2020']['Close'])

plt.title('Stocks in 2020')



plt.show()