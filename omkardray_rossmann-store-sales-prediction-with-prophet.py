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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

train.head()
test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

test.head()
store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

store.head()
store.shape
store.isnull().sum()
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
store.fillna(0, inplace = True)
store.head()
store.info()
store.PromoInterval.nunique()
store.PromoInterval.value_counts()
train.head()
train.isnull().sum()
train.describe()
train.head()
train.Date = pd.to_datetime(train.Date)
train.head()
train.head()
train.head()
train['year'] = pd.DatetimeIndex(train['Date']).year

train['month'] = pd.DatetimeIndex(train['Date']).month
train.head()
sns.barplot(x = 'month', y = 'Sales', data = train)

plt.show()
sns.barplot(x = 'year', y = 'Sales', data = train)

plt.show()
train.info()
train_store = pd.merge(train, store, how = 'inner', on = 'Store')



print("In total: ", train_store.shape)

train_store.head(10)
train_store.Store.nunique()
train_store.groupby('StoreType')['Sales'].describe()
train_store.groupby('StoreType')['Customers', 'Sales'].describe()

train_store.groupby('StoreType')['Customers'].describe()
train_store.head()
sns.catplot(data = train_store, x = 'month', y = "Sales", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               color = 'year') 
sns.factorplot(data = train_store, x = 'month', y = "Customers", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               ) 
sns.catplot(data = train_store, x = 'year', y = "Sales", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               color = 'year') 
sns.factorplot(data = train_store, x = 'month', y = "Sales", 

               col = 'DayOfWeek', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'StoreType', # per store type in rows

              ) 

train_store.groupby('DayOfWeek')['Sales'].describe()
train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].unique()
train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].nunique()
train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['StoreType'].value_counts()
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 

               col = 'Promo', 

               row = 'Promo2',

               hue = 'StoreType',

               palette = 'RdPu') 
train_store.head()
train_store.plot.line(x = 'Date', y = 'Sales')

plt.show()
sales = train_store[['Sales']]
sales.rolling(6).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)

plt.show()
train_store.info()
pd.plotting.autocorrelation_plot(train_store.head(10000)['Sales'])

plt.show()
train_store = train_store.set_index('month')
train_store.head()
train_store.tail()
store.head()
from fbprophet import Prophet
sales = train_store.rename(columns = {'Date': 'ds',

                                'Sales': 'y'})

sales.head()
sales=sales[['ds','y']]
sales.head()
ax = sales.set_index('ds').plot(figsize = (12, 4))

ax.set_ylabel('Daily Number of Sales')

ax.set_xlabel('Date')

plt.show()
train_store.head()
df= train_store
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b') & (df.StateHoliday == 'c')].loc[:, 'Date'].values

school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values



state = pd.DataFrame({'holiday': 'state_holiday',

                      'ds': pd.to_datetime(state_dates)})

school = pd.DataFrame({'holiday': 'school_holiday',

                      'ds': pd.to_datetime(school_dates)})



holidays = pd.concat((state, school))      

holidays
my_model = Prophet(interval_width = 0.95, 

                   holidays = holidays.head(50000))

my_model.fit(sales)



# dataframe that extends into future 6 weeks 

future_dates = my_model.make_future_dataframe(periods = 6*7)



print("First week to forecast.")

future_dates.tail(7)
forecast = my_model.predict(future_dates.head(10000))



# preditions for last week

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)