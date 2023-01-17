import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import *

import gc

sns.set()



PATH='/kaggle/input/ashrae-energy-prediction/'

train=pd.read_csv(PATH+'train.csv')

test=pd.read_csv(PATH+'test.csv')

weather_train=pd.read_csv(PATH+'weather_train.csv')

weather_test=pd.read_csv(PATH+'weather_test.csv')

building=pd.read_csv(PATH+'building_metadata.csv')
# merge the data

train = train.merge(building, on='building_id', how='left')

test = test.merge(building, on='building_id', how='left')

train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')

del weather_train, weather_test,building

gc.collect()
# check whether there are missing values

data_na=(train.isnull().sum()/len(train))*100

data_na=data_na.drop(data_na[data_na==0].index).sort_values(ascending=False)

missing_data=pd.DataFrame({'MissingRatio':data_na})

print(missing_data)
train.info()
# Saving the memory space

data_types = {'building_id': np.int16,

          'meter': np.int8,

          'site_id': np.int8,

          'square_feet': np.int32,

          'year_built': np.float16,

          'floor_count': np.float16,    

          'cloud_coverage': np.float16,

          'precip_depth_1_hr': np.float16,

           'wind_direction': np.float16,     

          'dew_temperature': np.float32,

          'air_temperature': np.float32,

          'sea_level_pressure': np.float32,

          'wind_speed': np.float32,

          'primary_use': 'category',}



for feature in data_types:

    train[feature] = train[feature].astype(data_types[feature])

    test[feature] = test[feature].astype(data_types[feature])

    

train["timestamp"] = pd.to_datetime(train["timestamp"])

test["timestamp"] = pd.to_datetime(test["timestamp"])

gc.collect();
train.head()
fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(x='primary_use', y='year_built', data=train)

plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(x='primary_use', y='square_feet', data=train)

plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(15,10))

sns.boxplot(x='primary_use', y='square_feet', hue='meter',data=train)

plt.xticks(rotation=90)
sample=pd.DataFrame(train,columns=['site_id','primary_use'])

sample.drop_duplicates(keep='first')
# We can see it by month

fig, axes = plt.subplots(8,2,figsize=(15, 30))

color_dic={'red':0,'blue':1,'orange':2,'purple':3}

for i in range(0,15):    

    for color,meter in color_dic.items():

        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):

            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by month', fontsize=13)

        axes[i%8][i//8].legend();

        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

        plt.subplots_adjust(hspace=0.45)
# We can also see it by week

fig, axes = plt.subplots(8,2,figsize=(15, 30))

color_dic={'red':0,'blue':1,'orange':2,'purple':3}

for i in range(0,15):    

    for color,meter in color_dic.items():

        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):

            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('W').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by week', fontsize=13)

        axes[i%8][i//8].legend();

        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

        plt.subplots_adjust(hspace=0.45)
# By day

fig, axes = plt.subplots(8,2,figsize=(15, 30))

color_dic={'red':0,'blue':1,'orange':2,'purple':3}

for i in range(0,15):    

    for color,meter in color_dic.items():

        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):

            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by day', fontsize=13)

        axes[i%8][i//8].legend();

        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

        plt.subplots_adjust(hspace=0.45)
# By hour

fig, axes = plt.subplots(8,2,figsize=(15, 30))

color_dic={'red':0,'blue':1,'orange':2,'purple':3}

for i in range(0,15):    

    for color,meter in color_dic.items():

        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):

            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by hour', fontsize=13)

        axes[i%8][i//8].legend();

        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

        plt.subplots_adjust(hspace=0.45)
# the weather condition(by day)

fig, axes = plt.subplots(figsize=(20,8))

axes1=axes.twinx()

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')

train[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes,alpha=0.9, label='air_temperature', color='red')

train[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes,alpha=0.9, label='dew_temperature', color='blue')

plt.legend()
# the cloud condition(by day)

fig, axes = plt.subplots(figsize=(20,8))

axes1=axes.twinx()

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')

train[['timestamp', 'cloud_coverage']].set_index('timestamp').resample('D').mean()['cloud_coverage'].plot(ax=axes,alpha=0.9, label='cloud_coverage', color='cyan')

plt.legend()
# the wind condition(by day)

fig, axes = plt.subplots(figsize=(20,8))

axes1=axes.twinx()

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')

train[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes,alpha=0.9, label='wind_speed', color='purple')

plt.legend()
for i in range(0,4):

    corr = train[train.meter == i][['timestamp','meter_reading','square_feet','year_built','floor_count',

             'air_temperature','cloud_coverage','dew_temperature','sea_level_pressure','wind_direction','wind_speed']].corr()

    f, ax = plt.subplots(figsize=(18, 6))

    sns.heatmap(corr,annot=True,cmap='RdGy')