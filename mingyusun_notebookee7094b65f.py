# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
building_meta_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
weather_train_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
print(train_df.shape)
print(building_meta_df.shape)
print(weather_train_df.shape)
print(train_df.isna().sum())
print(weather_train_df.isna().sum())
print(building_meta_df.isna().sum())
train_df.head()
weather_train_df.head()
building_meta_df.head()
train_df['timestamp']=pd.to_datetime(train_df['timestamp'],format='%Y-%m-%d %H:%M:%S')
train_electricity=train_df[train_df['meter']==0]

fig, axes = plt.subplots(1, 1, figsize=(14, 6))
train_electricity[train_electricity['building_id']==0][['timestamp','meter_reading']].set_index('timestamp').plot(ax=axes).set_ylabel('building 0 electricity/kWh')
train_electricity[train_electricity['building_id']==0][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean().plot(ax=axes).set_ylabel('building 0 electricity/kWh')
train_electricity[train_electricity['building_id']==0][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes).set_ylabel('building 0 electricity/kWh')
axes.set_title('building 0 electricity hourly, daily, and monthly meter reading')
axes.legend(['hourly','daily','monthly']) 
train_hotwater=train_df[train_df['meter']==3]
fig, axes = plt.subplots(1, 1, figsize=(14, 6))
train_hotwater[train_hotwater['building_id']==106][['timestamp','meter_reading']].set_index('timestamp').plot(ax=axes).set_ylabel('building 0 hotwater/kWh')
train_hotwater[train_hotwater['building_id']==106][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean().plot(ax=axes).set_ylabel('building 0 hotwater/kWh')
train_hotwater[train_hotwater['building_id']==106][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes).set_ylabel('building 0 hotwater/kWh')
axes.set_title('building 0 hotwater hourly, daily, and monthly meter reading')
axes.legend(['hourly','daily','monthly']) 
fig, axes = plt.subplots(2,2)
train_df[train_df['meter']==0].meter_reading.plot.hist(ax=axes[0,0])
train_df[train_df['meter']==1].meter_reading.plot.hist(ax=axes[0,1])
train_df[train_df['meter']==2].meter_reading.plot.hist(ax=axes[1,0])
train_df[train_df['meter']==3].meter_reading.plot.hist(ax=axes[1,1])
axes[0,0].set_title('electricity')
axes[0,1].set_title('chilledwater')
axes[1,0].set_title('steam')
axes[1,1].set_title('hotwater')
fig, axes = plt.subplots(1,4,figsize=(10,6))
train_df[train_df['meter']==0].meter_reading.plot.box(ax=axes[0])
train_df[train_df['meter']==1].meter_reading.plot.box(ax=axes[1])
train_df[train_df['meter']==2].meter_reading.plot.box(ax=axes[2])
train_df[train_df['meter']==3].meter_reading.plot.box(ax=axes[3])
axes[0].set_title('electricity')
axes[1].set_title('chilledwater')
axes[2].set_title('steam')
axes[3].set_title('hotwater')
lower_quantile_meter_reading=train_df['meter_reading'].describe()[4]
higher_quantile_meter_reading=train_df['meter_reading'].describe()[6]
plt.figure()
train_df[(train_df['meter_reading']>lower_quantile_meter_reading)&(train_df['meter_reading']<higher_quantile_meter_reading)].meter_reading.hist()
log_meter_reading=np.log1p(train_df['meter_reading']).plot.hist()
log_meter_reading.set_title('log transformed meter reading')
train_df['meter_reading']=np.log1p(train_df['meter_reading'])
fig, axes = plt.subplots(5, 1, figsize=(10, 10))
train_df[['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes[0])
train_df[train_df['meter']==0][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes[1])
train_df[train_df['meter']==1][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes[2])
train_df[train_df['meter']==2][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes[3])
train_df[train_df['meter']==3][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean().plot(ax=axes[4])
axes[0].legend(['all meter data'],loc='upper left')
axes[1].legend(['electricity meter data'],loc='upper left')
axes[2].legend(['chilledwater meter data'],loc='upper left')
axes[3].legend(['steam meter data'],loc='upper left')
axes[4].legend(['hotwater meter data'],loc='upper left')
axes[0].set_title('monthly meter reading trend')
hour=train_df.timestamp.dt.hour # extract 'hour' from datetime.
train_df['hour']=hour
fig, axes = plt.subplots(5,1,figsize=(10, 10))
train_df.groupby('hour').mean().meter_reading.plot(ax=axes[0])
train_df[train_df['meter']==0].groupby('hour').mean().meter_reading.plot(ax=axes[1])
train_df[train_df['meter']==1].groupby('hour').mean().meter_reading.plot(ax=axes[2])
train_df[train_df['meter']==2].groupby('hour').mean().meter_reading.plot(ax=axes[3])
train_df[train_df['meter']==3].groupby('hour').mean().meter_reading.plot(ax=axes[4])
axes[0].legend(['all meter data'],loc='upper left')
axes[1].legend(['electricity meter data'],loc='upper left')
axes[2].legend(['chilledwater meter data'],loc='upper left')
axes[3].legend(['steam meter data'],loc='upper left')
axes[4].legend(['hotwater meter data'],loc='upper left')
axes[0].set_title('hourly meter reading trend')
print(building_meta_df.primary_use.value_counts())
train_whole=train_df.merge(building_meta_df, on='building_id', how='left')
train_whole.head()
train_whole['timestamp']=pd.to_datetime(train_whole['timestamp'])
train_whole['month']=train_whole['timestamp'].dt.month
train_whole['meter_reading']=np.log1p(train_whole['meter_reading'])
fig, axes = plt.subplots(8, 2, figsize=(14, 16))
plt.subplots_adjust(hspace=0.8)
train_whole[['meter_reading','primary_use','month']].groupby(['month','primary_use']).sum().meter_reading.unstack().plot(subplots=True,ax=axes)
plt.legend(loc='upper right') 
sns.boxplot(x='meter_reading',y='primary_use',data=train_whole, orient='h')
(np.log1p(building_meta_df.square_feet)).hist()
building_meta_df.floor_count.value_counts()
building_meta_df.year_built.value_counts()
weather_train_df.air_temperature.dropna().hist() 
weather_train_df.dew_temperature.hist() 
