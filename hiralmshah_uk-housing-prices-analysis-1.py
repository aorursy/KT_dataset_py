import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import math
import datetime as datetime

%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
uk_housing_prices = pd.read_csv('../input/price_paid_records.csv')
# Any results you write to the current directory are saved as output.
uk_housing_prices.columns
uk_housing_prices.shape
uk_housing_prices['County'].value_counts()
berkshire_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'BERKSHIRE']
berkshire_housing_prices.shape
berkshire_housing_prices.head(10)
berkshire_housing_prices.describe()
berkshire_housing_prices.isnull().sum()
berkshire_housing_prices['Price'].hist(bins = 50)
berkshire_housing_prices.boxplot(column='Price')
berkshire_housing_prices.boxplot(column='Price', by = 'Old/New')
berkshire_housing_prices.boxplot(column = 'Price', by = 'Property Type')
berkshire_housing_prices.boxplot(column = 'Price', by ='Town/City', figsize = (16,8))
berkshire_housing_prices.boxplot(column = 'Price', by = 'Duration')
temp1 = pd.crosstab(berkshire_housing_prices['Town/City'], berkshire_housing_prices['Property Type'])
temp1.plot(kind = 'bar', stacked = True, color =['red','blue','green','orange'], grid = False)
temp2 = pd.crosstab(berkshire_housing_prices['Duration'], berkshire_housing_prices['Old/New'])
temp2.plot(kind='bar', stacked = True, color =['red','blue'], grid = False)
berkshire_housing_prices['Price_log'] = np.log(berkshire_housing_prices['Price'])
berkshire_housing_prices['Price_log'].hist(bins = 50)
berkshire_housing_prices['Date of Transfer'] = pd.to_datetime(berkshire_housing_prices['Date of Transfer'], format='%Y-%m-%d %H:%M')
berkshire_housing_prices.index = berkshire_housing_prices['Date of Transfer']
df = berkshire_housing_prices.loc[:,['Price']]
ts = df['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('House Price')
plt.legend(loc = 'best')
plt.show()
uk_housing_prices['Date of Transfer'] = pd.to_datetime(uk_housing_prices['Date of Transfer'], format='%Y-%m-%d %H:%M')

uk_housing_prices.dtypes
uk_housing_prices.index = uk_housing_prices['Date of Transfer']
uk_housing_prices.index
uk_housing_prices['year'] = pd.DatetimeIndex(uk_housing_prices['Date of Transfer']).year
uk_housing_prices['month'] = pd.DatetimeIndex(uk_housing_prices['Date of Transfer']).month

uk_housing_prices.head(5)
uk_housing_prices.groupby('year')['Price'].mean().plot.bar()
county_data = uk_housing_prices.groupby('County')['Price'].mean()
county_data.plot(figsize = (20,8), title = 'UK County House Prices')
temp = uk_housing_prices.groupby(['year','month'])['Price'].mean()
temp.plot(figsize=(16,5), title = 'UK Housing Price(Monthwise)', fontsize = 12)
temp3 = pd.crosstab(uk_housing_prices['Old/New'], uk_housing_prices['Property Type'])
temp3.plot(kind='bar', stacked=True, grid=False, figsize=(18,8))
temp = uk_housing_prices.groupby(['Old/New','Property Type'])['Price'].mean()
temp.plot(figsize=(16,5), title = 'UK Housing Price(Monthwise)', fontsize = 12)
london_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'GREATER LONDON']
london_housing_prices.shape
london_housing_prices.dtypes
london_housing_prices.index = london_housing_prices['Date of Transfer']
df1 = london_housing_prices.loc[:,['Price']]
ts1 = df1['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='London House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('London House Price')
plt.legend(loc = 'best')
plt.show()
machester_housing_prices = uk_housing_prices.loc[uk_housing_prices['County'] == 'GREATER MANCHESTER']
machester_housing_prices.shape
machester_housing_prices.index = machester_housing_prices['Date of Transfer']
df2 = machester_housing_prices.loc[:,['Price']]
ts2 = df2['Price']
plt.figure(figsize=(16,8))
plt.plot(ts, label='Machester House Price')
plt.title('Time series')
plt.xlabel('Time(year-month)')
plt.ylabel('Machester House Price')
plt.legend(loc = 'best')
plt.show()
