import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
df_io = pd.read_csv('../input/electricity-consumption/train.csv', date_parser='datetime')
df_io.head()
df_io.describe().T
df = (df_io
      .set_index(pd.to_datetime(df_io.datetime))
      .sort_index()
      .rename(columns = {'var1':'dewpnt', 'var2':'peak_demand'})
      .drop(columns = ['ID', 'datetime']))
df.isna().sum()
fig = plt.figure(figsize=(18, 9))
ax = fig.gca()
ax.plot(df.electricity_consumption)
ax.set_title('Total City Electricity Consumption ')
ax.set_ylabel('Power (kWatt)')
ax.set_xlabel('Datetime')
plt.show()   ## display figure plt
_ = df['electricity_consumption'].plot.hist(figsize=(13, 7), bins=200, title='Distribution of electricity_consumption Load')
plt.xlabel('Power (kWatt)')
plt.show()
# Creating new featheres from the dataset.
df = (df.assign(day_of_week = df.index.dayofweek,
                year = df.index.year,
                month = df.index.month,
                day = df.index.day, 
                week = df.index.week,
                week_day = df.index.weekday_name, 
                quarter = df.index.quarter,
                hour = df.index.hour))
df.head()
_ = (df['electricity_consumption'].loc[ (df['electricity_consumption'].index >= '2016-12-01') &
                                        (df['electricity_consumption'].index  < '2017-01-01') ] 
                                        .plot(figsize=(15, 5), title = 'December 2016'))
plt.show()
# Summer consumption
_ = (df['electricity_consumption'].loc[ (df['electricity_consumption'].index >= '2016-07-01') &
                                        (df['electricity_consumption'].index  < '2016-08-01') ] 
                                        .plot(figsize=(15, 5), title = 'July 2016'))
plt.show()
_ = df.pivot_table(index   = df['hour'], 
                   columns = 'week_day', 
                   values  = 'windspeed',
                   aggfunc = 'mean').plot(figsize=(18,6),
                   title   = 'windspeed - Daily Trends')
plt.ylabel('Windspeed')
plt.show()
_ = df.pivot_table(index   = df['hour'], 
                   columns = 'week_day', 
                   values  = 'pressure',
                   aggfunc = 'mean').plot(figsize=(18,6),
                   title   = 'pressure - Daily Trends')
plt.ylabel('Pressure')
plt.show()
_ = df.pivot_table(index   = df['hour'], 
                   columns = 'week_day', 
                   values  = 'temperature',
                   aggfunc = 'mean').plot(figsize=(18,6),
                   title   = 'temperature - Daily Trends')
plt.ylabel('Temperature (Celsius)')
plt.show()
_ = df.pivot_table(index   = df['hour'], 
                   columns = 'week_day', 
                   values  = 'dewpnt',
                   aggfunc = 'mean').plot(figsize=(18,6),
                   title   = 'dew point - Daily Trends')
plt.ylabel('Dew Point (Celsius)')
plt.show()
_ = df[['electricity_consumption','hour']].plot(x = 'hour',
                                                y ='electricity_consumption',
                                                kind = 'scatter',
                                                figsize = (17,6),
                                                title = 'Consumption by Hour of Day')
plt.ylabel('Power (kWatt)')
plt.show()
# %matplotlib notebook
_ = df.pivot_table(index   = df['hour'], 
                   columns = 'week_day', 
                   values  = 'electricity_consumption',
                   aggfunc = 'mean').plot(figsize=(18,6),
                   title   = 'electricity_consumption - Daily Trends')
plt.ylabel('Power (kWatt)')
plt.show()
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==1].hour, 
            df.loc[df['quarter']==1].electricity_consumption)
ax.set_title('Hourly Boxplot electricity_consumption Q1')
plt.ylabel('Power (kWatt)')

fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==2].hour,
            df.loc[df['quarter']==2].electricity_consumption)
ax.set_title('Hourly Boxplot electricity_consumption Q2')
plt.ylabel('Power (kWatt)')

fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==3].hour, 
            df.loc[df['quarter']==3].electricity_consumption)
ax.set_title('Hourly Boxplot electricity_consumption Q3')
plt.ylabel('Power (kWatt)')

fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==4].hour,
            df.loc[df['quarter']==4].electricity_consumption)
ax.set_title('Hourly Boxplot electricity_consumption Q4')
plt.ylabel('Power (kWatt)')

plt.show()
_ = df.pivot_table(index   = df['month'], 
                   columns = 'year', 
                   values  = 'electricity_consumption',
                   aggfunc = 'mean').plot(figsize = (18,6),
                                          title   = 'electricity_consumption - Yearly Trends')
plt.ylabel('Power (kWatt)')
plt.show()
# From july to aug we have the largest consumption.
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.month, 
            df.electricity_consumption)
ax.set_title('Monthly Boxplot electricity_consumption')
plt.ylabel('Power (kWatt)')

plt.show()
_ = df.pivot_table(index = df['month'], 
                   columns = 'year', 
                   values  = 'pressure',
                   aggfunc = 'mean').plot(figsize = (18,6),
                                          title   = 'pressure - Yearly Trends')
plt.ylabel('Pressure')
plt.show()
_ = df.pivot_table(index   = df['month'], 
                   columns = 'year', 
                   values  = 'windspeed',
                   aggfunc = 'mean').plot(figsize = (18,6),
                                          title   = 'windspeed - Yearly Trends')
plt.ylabel('Windspeed')
plt.show()
_ = df.pivot_table(index   = df['month'], 
                   columns = 'year', 
                   values  = 'temperature',
                   aggfunc = 'mean').plot(figsize = (18,6),
                                          title   = 'temperature - Yearly Trends')
plt.ylabel('Temperature (Celsius)')
plt.show()
_ = df.pivot_table(index   = df['month'], 
                   columns = 'year', 
                   values  = 'dewpnt',
                   aggfunc = 'mean').plot(figsize = (18,6),
                                          title   = 'dew point - Yearly Trends')
plt.ylabel('Dew Point (Celsius)')
plt.show()
fig, ax = plt.subplots(figsize = (17,8))
corr = df.corr()
ax = sns.heatmap(corr, annot=True,
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)
plt.show()
from IPython.display import Image
Image("../input/tarif-2010/tarif_2010.jpg")