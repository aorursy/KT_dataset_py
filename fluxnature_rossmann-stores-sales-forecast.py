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
import numpy as ny 
import seaborn as sns 
import matplotlib.pyplot as plt
import datetime 
sales_train_df = pd.read_csv("../input/sales-train-store-dataset/train.csv")
sales_train_df.head()
sales_train_df.tail()
sales_train_df.info()
sales_train_df.describe()
store_info_df = pd.read_csv("../input/sales-train-store-dataset/store.csv")
store_info_df.head()
store_info_df.tail()
store_info_df.describe()
store_info_df.info()
sns.heatmap(sales_train_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues' )
sales_train_df.hist(bins = 30, figsize = (20, 20), color = 'green')
sales_train_df['Customers'].max()
open_train_df = sales_train_df[sales_train_df['Open'] == 1]
closed_train_df = sales_train_df[sales_train_df['Open'] == 0]
print('total', len(sales_train_df))
print('number of closed stores', len(closed_train_df))
print('number of opened stores', len(open_train_df))
sales_train_df = sales_train_df[sales_train_df['Open'] == 1]
sales_train_df
sales_train_df.drop(['Open'], axis = 1, inplace = True)
sales_train_df
sales_train_df.describe()
sns.heatmap(store_info_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues' )
store_info_df[store_info_df['CompetitionDistance'].isnull()]
store_info_df[store_info_df['CompetitionOpenSinceMonth'].isnull()]
store_info_df[store_info_df['Promo2'] == 0]
str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']
for str in str_cols:
    store_info_df[str].fillna(0, inplace = True)
sns.heatmap(store_info_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues' )
store_info_df['CompetitionDistance'].fillna(store_info_df['CompetitionDistance'].mean(), inplace = True)
sns.heatmap(store_info_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues' )
store_info_df.hist(bins = 30, figsize = (20, 20), color = 'r')
sales_train_all_df = pd.merge(sales_train_df, store_info_df, how = 'inner', on = 'Store')
sales_train_all_df.head()
correlations = sales_train_all_df.corr()['Sales'].sort_values()
correlations
correlations = sales_train_all_df.corr()
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(correlations, annot = True)
sales_train_all_df['Year'] = pd.DatetimeIndex(sales_train_all_df['Date']).year
sales_train_all_df
sales_train_all_df['Month'] = pd.DatetimeIndex(sales_train_all_df['Date']).month
sales_train_all_df['Day'] = pd.DatetimeIndex(sales_train_all_df['Date']).day
sales_train_all_df
axis = sales_train_all_df.groupby('Month')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'red')
axis.set_title('Average sales per month')

plt.figure()

axis = sales_train_all_df.groupby('Month')[['Customers']].mean().plot(figsize = (10,5), marker = 'o', color = 'b')
axis.set_title('Average customers per month')
axis = sales_train_all_df.groupby('Day')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'red')
axis.set_title('Average sales per day')

plt.figure()

axis = sales_train_all_df.groupby('Day')[['Customers']].mean().plot(figsize = (10,5), marker = 'o', color = 'b')
axis.set_title('Average customers per day')
axis = sales_train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'red')
axis.set_title('Average sales per dayofweek')

plt.figure()

axis = sales_train_all_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = (10,5), marker = 'o', color = 'b')
axis.set_title('Average customers per dayofweek')
fig, ax = plt.subplots(figsize = (20,20))
sales_train_all_df.groupby(['Date', 'StoreType']).mean()['Sales'].unstack().plot(ax = ax)
plt.figure(figsize = [15,10])

plt.subplot(211)
sns.barplot(x = 'Promo', y = 'Sales', data = sales_train_all_df)

plt.subplot(212)
sns.barplot(x = 'Promo', y = 'Customers', data = sales_train_all_df)
plt.figure(figsize = [15,10])

plt.subplot(211)
sns.violinplot(x = 'Promo', y = 'Sales', data = sales_train_all_df)

plt.subplot(212)
sns.violinplot(x = 'Promo', y = 'Customers', data = sales_train_all_df)
from fbprophet import Prophet
from fbprophet import Prophet
def sales_predictions(Store_ID, sales_df, periods):
    
    sales_df = sales_df[sales_df['Store'] == Store_ID]
    sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values('ds')
    
    model = Prophet()
    model.fit(sales_df)
    future = model.make_future_dataframe(periods = periods)
    
    forecast = model.predict(future)
    figure = model.plot(forecast, xlabel = 'Date', ylabel = 'Sales')
    figure2 = model.plot_components(forecast)
    
sales_predictions(10, sales_train_all_df, 60)
def sales_predictions(Store_ID, sales_df, holidays, periods):
    
    sales_df = sales_df[sales_df['Store'] == Store_ID]
    sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values('ds')
    
    model = Prophet(holidays = holidays)
    model.fit(sales_df)
    future = model.make_future_dataframe(periods = periods)
    
    forecast = model.predict(future)
    figure = model.plot(forecast, xlabel = 'Date', ylabel = 'Sales')
    figure2 = model.plot_components(forecast)
    
school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1].loc[:, 'Date'].values
school_holidays.shape
state_holidays = sales_train_all_df[(sales_train_all_df['StateHoliday'] == 'a') | (sales_train_all_df['StateHoliday'] == 'b') | (sales_train_all_df['StateHoliday'] == 'c')].loc[:, 'Date'].values
state_holidays.shape
state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                              'holiday': 'state_holidays'})
state_holidays 
school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                              'holiday': 'school_holidays'})
school_holidays
school_state_holidays = pd.concat((state_holidays, school_holidays))
school_state_holidays
sales_predictions(6, sales_train_all_df, school_state_holidays, 90)
