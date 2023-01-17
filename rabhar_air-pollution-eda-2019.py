import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("/kaggle/input/air-pollution-dataset-india-2019/combined.csv", low_memory=False)
df.head()
def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)
df['datetime'] = lookup(df['datetime'])
#df.set_index('datetime', inplace=True)
df.head()
df.info(memory_usage='deep')
df.describe()
df.drop('live', axis=1, inplace=True) #dropping unwanted column
df.shape
print(df['SO2'].isna().sum())
print(df['NO2'].isna().sum())
print(df['NH3'].isna().sum())
print(df['CO'].isna().sum())
print(df['OZONE'].isna().sum())
#mean of the day
dayMeandf =  df.loc[:,['id','SO2','NO2','NH3','CO','OZONE']].groupby([df['datetime'].dt.dayofyear ,'id']).transform('mean')
df['SO2'].fillna(dayMeandf['SO2'], inplace=True)
df['NO2'].fillna(dayMeandf['NO2'], inplace=True)
df['NH3'].fillna(dayMeandf['NH3'], inplace=True)
df['CO'].fillna(dayMeandf['CO'], inplace=True)
df['OZONE'].fillna(dayMeandf['OZONE'], inplace=True)
#mean of the week
weekMeandf =  df.loc[:,['id','SO2','NO2','NH3','CO','OZONE']].groupby([df['datetime'].dt.weekofyear ,'id']).transform('mean')
df['SO2'].fillna(weekMeandf['SO2'], inplace=True)
df['NO2'].fillna(weekMeandf['NO2'], inplace=True)
df['NH3'].fillna(weekMeandf['NH3'], inplace=True)
df['CO'].fillna(weekMeandf['CO'], inplace=True)
df['OZONE'].fillna(weekMeandf['OZONE'], inplace=True)
print(df['SO2'].isna().sum())
print(df['NO2'].isna().sum())
print(df['NH3'].isna().sum())
print(df['CO'].isna().sum())
print(df['OZONE'].isna().sum())
df.dropna(how='all', inplace=True)
#number of state
df['stateid'].unique()
def myfunc(x):
    return (x['id'].nunique())

stationsPerState = df.loc[:,['stateid','id']].groupby('stateid').apply(myfunc).reset_index()
stationsPerState.rename({0:'count'}, axis=1, inplace=True)
stationsPerState.sort_values('count', ascending=False, inplace=True)
stationsPerState
fig = plt.figure(figsize=[10,4])
axes = fig.add_axes([0,0,1,1])
axes.bar(stationsPerState['stateid'], stationsPerState['count'])
fig.autofmt_xdate(rotation=60)
axes.set_xlabel('State',{"fontsize":12})
axes.set_ylabel('Number of stations', {"fontsize":12})
axes.set_title('Number of stations per state')
def myfunc(x):
    return (x['id'].nunique())

stationsByState = df.loc[:,['stateid','cityid','id']].groupby(['stateid','cityid']).apply(myfunc)
stationsByState
fig = plt.figure(figsize=[20,20])
axes = fig.subplots(7,3)
axes = axes.flatten()
fig.tight_layout()
fig.subplots_adjust(hspace=1, wspace=0.2)

for i,state in enumerate(stationsByState.index.get_level_values(0).unique()):
    axes[i].bar(stationsByState[state].index,stationsByState[state].values)
    axes[i].tick_params(axis='x', labelrotation=45)
    axes[i].set_xlabel(state,{"fontsize":14})
    axes[i].set_ylabel("stations", {"fontsize":14})

averageByState = df.loc[:,['stateid','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('stateid', as_index=False).mean()
averageByState
fig = plt.figure(figsize=[20,50])
pollutants = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']
colors = ['#00965d','#829600', '#dbeb34', '#34ebd3', '#004496','#7a0096','#960052']
axes = fig.subplots(len(pollutants))
fig.subplots_adjust(hspace=0.4)

for i,pollutant in enumerate(pollutants):
    axes[i].bar(averageByState['stateid'], averageByState[pollutant], color=colors[i])
    axes[i].tick_params(axis='x', labelrotation=45)
    axes[i].set_xlabel('States', {"fontsize":16})
    axes[i].set_ylabel('value', {"fontsize":16})
    axes[i].set_title(pollutant, {"fontsize":18})
fig = plt.figure(figsize=[20,6])
axes = fig.add_axes([0,0,1,1])
axes.plot(averageByState['stateid'], averageByState.loc[:,['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']])
axes.tick_params(axis='x', labelrotation=45)
axes.legend(labels=['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],prop={'size': 18})
axes.set_xlabel('State', {"fontsize":18})
axes.set_ylabel('Pollutant value', {"fontsize":18})
averageByState.sort_values(['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'], ascending=False).head()
averageByState.sort_values(['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'], ascending=True).head()
df['month'] = df['datetime'].dt.month_name()
df['month'] = df['month'].astype(pd.CategoricalDtype(['January', 'February', 'March', 'April', 'May', 'June', 'July' ,'August', 'September', 'October', 'November', 'December'], ordered=True))
monthlyAverage = df.loc[:,['month','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('month', as_index=False).mean()
monthlyAverage
fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.scatter(monthlyAverage['month'],monthlyAverage['PM2.5'], label = 'PM2.5')
axes.scatter(monthlyAverage['month'],monthlyAverage['PM10'], label= 'PM10')
axes.scatter(monthlyAverage['month'],monthlyAverage['SO2'], label='SO2')
axes.scatter(monthlyAverage['month'],monthlyAverage['NO2'], label='NO2')
axes.scatter(monthlyAverage['month'],monthlyAverage['NH3'], label='NH3')
axes.scatter(monthlyAverage['month'],monthlyAverage['CO'], label= 'CO')
axes.scatter(monthlyAverage['month'],monthlyAverage['OZONE'], label='OZONE')
axes.legend()
df['hour'] = df['datetime'].dt.hour
df['hour'] = df['hour'].astype(pd.CategoricalDtype([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23], ordered=True))
hourlyAverage = df.loc[:,['hour','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('hour', as_index=False).mean()
fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.scatter(hourlyAverage['hour'],hourlyAverage['PM2.5'], label = 'PM2.5')
axes.scatter(hourlyAverage['hour'],hourlyAverage['PM10'], label= 'PM10')
axes.scatter(hourlyAverage['hour'],hourlyAverage['SO2'], label='SO2')
axes.scatter(hourlyAverage['hour'],hourlyAverage['NO2'], label='NO2')
axes.scatter(hourlyAverage['hour'],hourlyAverage['NH3'], label='NH3')
axes.scatter(hourlyAverage['hour'],hourlyAverage['CO'], label= 'CO')
axes.scatter(hourlyAverage['hour'],hourlyAverage['OZONE'], label='OZONE')
axes.set_xticks(range(24))
axes.set_xlabel('hours', {"fontsize":16})
axes.set_ylabel('value', {"fontsize":16})
axes.set_title('Pollution by Hour', {"fontsize":18})
axes.legend()
df['dayoftheweek'] = df['datetime'].dt.dayofweek
df['weekend'] = 'weekday'
df.loc[df['dayoftheweek'].isin([5,6]),'weekend'] = 'weekend'
df['weekend'] = df['weekend'].astype(pd.CategoricalDtype(['weekday','weekend']))

weekendGrp = df.loc[:,['weekend','PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].groupby('weekend', as_index=False).mean()
weekendGrp
weekendGrp[weekendGrp['weekend'] == 'weekend'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0]
fig = plt.figure(figsize=[20,6])
axes = fig.subplots()
axes.plot(['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],weekendGrp[weekendGrp['weekend'] == 'weekend'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0], label='weekend')
axes.plot(['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE'],weekendGrp[weekendGrp['weekend'] == 'weekday'].loc[:,['PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].values[0], label='weekday')
axes.legend(prop={'size':18})
axes.set_xlabel('Pollutant', {"fontsize":18})
axes.set_ylabel('value', {"fontsize":18})
axes.set_title('Pollution on weekdays vs weekends')
df[df['datetime'] == '2019-10-27'].mean()
df[df['datetime'] == '2019-01-01'].mean()
