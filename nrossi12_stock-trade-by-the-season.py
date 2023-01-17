# Import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

from sklearn import linear_model
# Import data file and assign to variable

stock_data = pd.read_csv('../input/sandp500/all_stocks_5yr.csv', parse_dates=['date'])
# Show all column names within dataset

stock_data.columns
# Start of data set

stock_data.head()
# End of dataset

stock_data.tail()
# Describe the data by column

stock_data.describe()
# Sum NaN values per column within dataset

stock_data.isnull().sum()
# Clean data set and ignore any rows with NaN value

stock_data = stock_data.dropna(how='any')

print(f"Shape of dataset is {stock_data.shape}")
# Good to have an idea of the date range

print(f"First recorded date: {stock_data['date'].min()}")

print(f"Last recorded date: {stock_data['date'].max()}")
year_data = stock_data.set_index('date').groupby(pd.Grouper(freq='Y'))

year_data = year_data['volume'].mean().plot(kind='bar')

year_data.set_xticklabels(('2013', '2014', '2015', '2016', '2017', '2018'))

year_data.set_ylabel('Average Volume')

year_data.set_xlabel('Year')

year_data.set_title('Average Volume traded per year')

plt.show()
#Retreiving the mean volume group by the month

avg_permonth = stock_data.set_index('date').groupby(pd.Grouper(freq='M'))

avg_permonth = avg_permonth['volume'].mean()
# Plot to show mean of volume per month of year

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

for i, (year, sg) in enumerate(avg_permonth.groupby(avg_permonth.index.year)):

    sg.plot(ax=axs[i//2, i%2])

    

fig.suptitle('Average volume traded per month of the year', fontsize=12)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
# Average volume per day

avg_perday = stock_data.set_index('date').groupby(pd.Grouper(freq='d'))

avg_perday['volume'].mean().plot(figsize=(15, 7))



# Average volume per week

avg_perweek = stock_data.set_index('date').groupby(pd.Grouper(freq='w'))

avg_perweek['volume'].mean().plot(figsize=(15, 7))



# Average volume per month

avg_permonth = stock_data.set_index('date').groupby(pd.Grouper(freq='m'))

avg_permonth['volume'].mean().plot(figsize=(15, 7))



plt.legend(('Daily mean', 'Weekly mean', 'Monthly mean'))

plt.title('Daily, Weekly, Monthly mean volume throughout data')

plt.xlabel('Date')

plt.ylabel('Volume')



plt.show()
# Apple 5 year history

aapl = stock_data.loc[stock_data.Name=='AAPL', :]

aapl
# Amazon 5 year history

msft = stock_data.loc[stock_data.Name=='MSFT', :]

msft
fig, axes = plt.subplots(1, 2, figsize=(20, 5))



# Apple graph

plt.subplot(121)

plt.plot(aapl['date'], aapl['open'])

plt.plot(aapl['date'], aapl['close'])

plt.title('Apple opening and closing prices')

plt.xlabel('Date')

plt.ylabel('Price')

plt.legend(('Open', 'Close'), loc='upper left')

plt.grid(True)



# Microsoft graph

plt.subplot(122)

plt.plot(msft['date'], msft['open'])

plt.plot(msft['date'], msft['close'])

plt.title('Microsoft opening and closing prices')

plt.xlabel('Date')

plt.ylabel('Price')

plt.grid(True)

plt.legend(('Open', 'Close'), loc='upper left')





plt.show()
fig, axes = plt.subplots(1, 2, figsize=(20, 5))



# Opening prices

plt.subplot(121)

plt.plot(aapl['date'], aapl['open'], '--b')

plt.plot(msft['date'], msft['open'], ':r')

plt.title('Opening prices')

plt.xlabel('Date')

plt.ylabel('Price')

plt.legend(('AAPL', 'MSFT'), loc='upper left')

plt.grid(True)



# Closing prices

plt.subplot(122)

plt.plot(aapl['date'], aapl['close'], '--b')

plt.plot(msft['date'], msft['close'], ':r')

plt.title('Closing prices')

plt.xlabel('Date')

plt.ylabel('Price')

plt.grid(True)

plt.legend(('AAPL', 'MSFT'), loc='upper left')





plt.show()
df1 = aapl.set_index('date').loc[:, ['low']]

df2 = aapl.set_index('date').loc[:, ['close']]

df3 = msft.set_index('date').loc[:, ['low']]

df4 = msft.set_index('date').loc[:, ['close']]



fig, axes = plt.subplots(1, 2, figsize=(20, 5))



plt.subplot(121)

plt.plot(df1.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df2.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('Low', 'Close'))

plt.title('AAPL weekly mean of low and closing price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.subplot(122)

plt.plot(df3.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df4.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('Low', 'Close'))

plt.title('MSFT weekly mean of low and closing price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.show()
df1 = aapl.set_index('date').loc[:, ['high']]

df2 = aapl.set_index('date').loc[:, ['open']]

df3 = msft.set_index('date').loc[:, ['high']]

df4 = msft.set_index('date').loc[:, ['open']]



fig, axes = plt.subplots(1, 2, figsize=(20, 5))



plt.subplot(121)

plt.plot(df1.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df2.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('High', 'Open'))

plt.title('AAPL weekly mean of high and opening price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.subplot(122)

plt.plot(df3.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df4.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('High', 'Open'))

plt.title('MSFT weekly mean of high and opening price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.show()
df1 = aapl.set_index('date').loc[:, ['high']]

df2 = aapl.set_index('date').loc[:, ['low']]

df3 = msft.set_index('date').loc[:, ['high']]

df4 = msft.set_index('date').loc[:, ['low']]



fig, axes = plt.subplots(1, 2, figsize=(20, 5))



plt.subplot(121)

plt.plot(df1.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df2.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('High', 'Low'))

plt.title('AAPL weekly mean of high and Low price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.subplot(122)

plt.plot(df3.groupby(pd.Grouper(freq='w')).mean())

plt.plot(df4.groupby(pd.Grouper(freq='w')).mean())

plt.grid(True)

plt.legend(('High', 'Low'))

plt.title('MSFT weekly mean of high and low price')

plt.xlabel('Date')

plt.ylabel('Price')



plt.show()
sns.pairplot(aapl, x_vars=['open', 'high', 'low', 'close'], y_vars=['volume'], size=10, kind='reg')
sns.pairplot(msft, x_vars=['open', 'high', 'low', 'close'], y_vars=['volume'], size=10, kind='reg')
aapl_2015 = aapl.set_index('date')

aapl_2015 = aapl_2015.loc['2015-01':'2015-12']
sns.pairplot(aapl_2015, x_vars=['open', 'high', 'low', 'close'], y_vars=['volume'], size=10, kind='reg')
input = 125

print(aapl.iloc[(aapl['open']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['high']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['low']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['close']-input).abs().argsort()[:2]])
aapl_2016 = aapl.set_index('date')

aapl_2016 = aapl_2016.loc['2016-01':'2016-12']
sns.pairplot(aapl_2016, x_vars=['open', 'high', 'low', 'close'], y_vars=['volume'], size=10, kind='reg')
input = 110

print(aapl.iloc[(aapl['open']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['high']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['low']-input).abs().argsort()[:2]])

print(aapl.iloc[(aapl['close']-input).abs().argsort()[:2]])
msft_2015 = msft.set_index('date')

msft_2015 = msft_2015.loc['2015-01':'2015-12']



sns.pairplot(msft_2015, x_vars=['open', 'close', 'low', 'high'], y_vars=['volume'], size=10, kind='reg')
input = 47

print(msft.iloc[(aapl['open']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['high']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['low']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['close']-input).abs().argsort()[:2]])
msft_2016 = msft.set_index('date')

msft_2016 = msft_2016.loc['2015-01':'2015-12']



sns.pairplot(msft_2016, x_vars=['open', 'close', 'low', 'high'], y_vars=['volume'], size=10, kind='reg')
input = 47

print(msft.iloc[(aapl['open']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['high']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['low']-input).abs().argsort()[:2]])

print(msft.iloc[(aapl['close']-input).abs().argsort()[:2]])
x = aapl[['open', 'low', 'high']]

y = aapl['close']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(len(X_train))

print(len(X_test))

print(len(y_train))

print(len(y_test))

print(f"Total records : {len(X_train) + len(X_test)}")
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)



print(round(linreg.intercept_, 3))

print(linreg.coef_)
linreg.predict(X_test)
y_test
round(linreg.score(X_test, y_test)*100, 3)