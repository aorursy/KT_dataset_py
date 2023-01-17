import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline
df = pd.read_csv("/kaggle/input/air-pollution-dataset-2020/combined.csv")
df = df[df['id'] == 'site_1422']
def lookup(s):

    dates = {date:pd.to_datetime(date) for date in s.unique()}

    return s.map(dates)
df['datetime'] = lookup(df['datetime'])
df.set_index('datetime', inplace=True)
df.head()
df.info(memory_usage=True)
df.describe()
df.drop('live', axis=1, inplace=True) #dropping unwanted column
df = df.resample('H').interpolate()
df['PM10'].fillna(500,inplace=True)
df.shape
fig = plt.figure(figsize=[20,10])

axes = fig.subplots()

axes.scatter(df.index, df['PM2.5'], label = 'PM2.5')

axes.scatter(df.index, df['PM10'], label = 'PM10')

axes.scatter(df.index, df['NO2'], label = 'NO2')

axes.scatter(df.index, df['SO2'], label = 'S02')

axes.scatter(df.index, df['NH3'], label = 'NH3')

axes.scatter(df.index, df['CO'], label = 'CO')

axes.scatter(df.index, df['OZONE'], label = 'OZONE')

axes.axvline(x=datetime.datetime(2020,3,25), color='r', label = 'Phase 1')

axes.axvline(x= datetime.datetime(2020,4,15), color='g', label = 'Phase 2')

axes.axvline(x= datetime.datetime(2020,5,4), color='b', label = 'Phase 3')

axes.axvline(x= datetime.datetime(2020,5,18), color='c', label = 'Phase 4')

axes.text(s = 'Before Lockdown', x = datetime.datetime(2020,2,1), y = 540, fontdict={'fontsize':18})

axes.text(s = 'After Lockdown', x = datetime.datetime(2020,4,20), y = 540, fontdict={'fontsize':18})

axes.text(s = 'Phase 1', x = datetime.datetime(2020,4,3), y = 420, fontdict={'fontsize':18, 'rotation':90})

axes.text(s = 'Phase 2', x = datetime.datetime(2020,4,25), y = 420, fontdict={'fontsize':18, 'rotation':90})

axes.text(s = 'Phase 3', x = datetime.datetime(2020,5,10), y = 420, fontdict={'fontsize':18, 'rotation':90})

axes.text(s = 'Phase 4', x = datetime.datetime(2020,5,19), y = 420, fontdict={'fontsize':18, 'rotation':90})

axes.set_xlabel('Date', fontsize=18)

axes.set_ylabel('Pollutant value', fontsize=18)

axes.legend(bbox_to_anchor=(1,1), fontsize=12)
df['lockdown'] = 'before_lockdown'

df.loc['2020-03-25':, 'lockdown'] = 'after_lockdown'
beforeLockdown = df.loc[df['lockdown'] == 'before_lockdown', ['PM2.5','PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].mean().reset_index()

beforeLockdown['lockdown'] = 'before'

beforeLockdown.rename({'index' : 'pollutant', 0 :'value' }, axis=1, inplace=True)

beforeLockdown
afterLockdown = df.loc[df['lockdown'] == 'after_lockdown', ['PM2.5','PM10', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']].mean().reset_index()

afterLockdown['lockdown'] = 'after'

afterLockdown.rename({'index' : 'pollutant', 0 :'value' }, axis=1, inplace=True)

afterLockdown
lockdownDf = pd.concat([beforeLockdown, afterLockdown], ignore_index=True)

lockdownDf
fig = plt.figure(figsize=[10,8])

ax = fig.subplots()

sns.barplot(x='pollutant', y='value',  hue='lockdown', data=lockdownDf, ax=ax)

ax.set_xlabel('Pollutant', fontsize=16)

ax.set_ylabel('Pollutant value', fontsize=16)

plt.show()