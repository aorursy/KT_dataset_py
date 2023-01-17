import sqlite3

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import datetime as dt





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
file = sqlite3.connect('../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite')
data = pd.read_sql_query("SELECT * FROM fires", con= file)
data.describe()
data['DATE'] = pd.to_datetime(data['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')

print(data.head())
# Converting to day of week

data['day_of_week'] = data['DATE'].dt.day_name()

print(data['day_of_week'])



# Converting to month of year

data['month'] = pd.DatetimeIndex(data['DATE']).month

data['month'] = data.month.replace([1,2,3,4,5,6,7,8,9,10,11,12], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

print(data['month'])
for col in data.columns:

    print(col)
lst = ['DATE', 'day_of_week', 'month', 'FIRE_YEAR', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR', 'STATE', 'FIRE_SIZE']

data_final = data[lst]

data_final.head()
dow = data_final.groupby(['day_of_week']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(14,5))

g = sns.barplot(data = dow, y = 'count', x = 'day_of_week')

plt.xlabel('Day of week')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing number of wildfire per day of week',fontsize=20)
arson_data_per_day = data[data['STAT_CAUSE_DESCR'] == 'Arson'].groupby(['day_of_week']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(14,5))

g = sns.barplot(data = arson_data_per_day, y = 'count', x = 'day_of_week')

plt.xlabel('Day of week')

plt.ylabel('Number of wildfire cases due to arson')

g.axes.set_title('Graph showing number of wildfire per day of week caused by arson',fontsize=20)
month_data = data_final.groupby(['month']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(14,5))

g = sns.barplot(data = month_data, y = 'count', x = 'month')

plt.xlabel('Month')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing number of wildfire per month',fontsize=20)
year_data = data_final.groupby(['FIRE_YEAR']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(14,5))

g = sns.barplot(data = year_data, y = 'count', x = 'FIRE_YEAR')

plt.xlabel('Year')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing number of wildfire per year',fontsize=20)
data_2006 = data[data['FIRE_YEAR'] == 2006].groupby(['STAT_CAUSE_DESCR']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(25,10))

g = sns.barplot(data = data_2006, y = 'count', x = 'STAT_CAUSE_DESCR')

plt.xlabel('Cause of wildfire')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing cause of wildfire in 2006',fontsize=20)
data_final['new_time'] = data_final['DISCOVERY_TIME'].astype(str).str[:2]
hour_data = data_final.groupby(['new_time']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(14,5))

g = sns.barplot(data = hour_data, y = 'count', x = 'new_time')

plt.xlabel('Hour')

plt.ylabel('Number of wildfire cases per hour in a day')

g.axes.set_title('Graph showing number of wildfire per hour in a day',fontsize=20)
cause_data = data_final.groupby(['STAT_CAUSE_DESCR']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(25,10))

g = sns.barplot(data = cause_data, y = 'count', x = 'STAT_CAUSE_DESCR')

plt.xlabel('Causes')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing number of wildfire per cause',fontsize=20)
# Creating barplot

plt.figure(figsize=(25,10))

g = sns.barplot(data = data_final, y = 'FIRE_SIZE', x = 'STAT_CAUSE_DESCR')

plt.xlabel('Causes')

plt.ylabel('Size of fire (acres)')

g.axes.set_title('Graph showing size of wildfire per cause',fontsize=20)
state_data = data_final.groupby(['STATE']).size().reset_index(name = 'count').sort_values('count')



# Creating barplot

plt.figure(figsize=(18,5))

g = sns.barplot(data = state_data, y = 'count', x = 'STATE')

plt.xlabel('State')

plt.ylabel('Number of wildfire cases')

g.axes.set_title('Graph showing number of wildfire per state',fontsize=20)