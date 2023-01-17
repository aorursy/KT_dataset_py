# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import calendar





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/denpasarbalihistoricalweatherdata/openweatherdata-denpasar-1990-2020v0.1.csv')
df.columns
df.head(3)
df.tail(3)
df.shape
df.info()
df.isnull().sum()
df.drop(['timezone', 'city_name', 'lat', 'lon','pressure', 'rain_1h',

       'rain_3h', 'rain_6h', 'rain_12h', 'rain_24h', 'rain_today', 'snow_1h',

       'snow_3h', 'snow_6h', 'snow_12h', 'snow_24h', 'snow_today',

       'clouds_all', 'weather_id','weather_icon'], axis=1, inplace=True)
df['date'] = pd.to_datetime(df['dt_iso'], infer_datetime_format=True)
df.info()
df.drop(['dt_iso'], axis=1, inplace=True)
df = df.set_index('date')
df['year'] = df.index.year

df['month'] = df.index.month

df['day'] = df.index.day

df['weekday_name'] = df.index.day_name()

df['hour'] = df.index.hour
plt.style.use('seaborn-whitegrid')
month_list = calendar.month_name[1:13]

month_list
df['1990':'2000'].groupby(['year','month'])['temp'].mean().unstack(0).plot(figsize=(16,10)).legend(bbox_to_anchor=(1,0.5))

plt.ylabel('Average Temp in C')

plt.xticks(ticks=df['month'].unique() ,labels = month_list)

plt.show()
def monthly_measurement(year, month, measurement):

    pvt_tbl = df['{}-{}'.format(year,month)].pivot_table(index='day', columns='hour', values=str(measurement)) #create pivot table

    fig, ax = plt.subplots(figsize=(12,10))

    

    month_name = calendar.month_name[month]

    

    sns.heatmap(pvt_tbl, cmap='coolwarm', ax=ax,linewidths=0.2)

    ax.set_title('Temperature in {} {}'.format(month_name, year))

    plt.show()
monthly_measurement(2019, 1, 'temp')
fig, ax = plt.subplots(figsize=(12,8))

sns.boxplot(x='month', y='temp', data=df['1990'], ax=ax)

ax.set_title('Box')

plt.show()
def percentage_of_monthly_rain(year, weather):

    

    temp = df[str(year)].groupby(['year', 'month','weather_main']).size().unstack(fill_value=0)

    total = np.sum(temp.iloc[:,:].values, axis=1)

    return ((temp.iloc[:,-1]+temp.iloc[:,-2])/total * 100).values
plt.figure(figsize=(16,8))

year = [1990, 1995, 2000, 2005, 2010, 2015]

for i in (year):

    num = percentage_of_monthly_rain(i, 'RAIN')

    plt.plot(month_list, num, label=str(i))

plt.xlabel('Month')

plt.ylabel('% Monthly Rain Occurrence')

plt.legend(bbox_to_anchor=(1,0.7))

plt.show()