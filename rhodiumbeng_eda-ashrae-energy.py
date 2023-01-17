import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sns.set()
# load data from csv files

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
print('Shape of the data:','\n','  train_csv: ',train.shape,'\n','  test_csv: ',test.shape,'\n',

      '  building_metadata.csv: ',building.shape,'\n','  weather_train.csv: ',weather_train.shape,'\n',

      '  weather_test.csv: ',weather_test.shape)
# look at the train_data

train.head()
# convert 'timestamp' which is a string into datetime object

train['timestamp'] = pd.to_datetime(train['timestamp'])



# let's also extract the pertinent date features like 'hour', 'day', 'day of week' and 'month'

train["hour"] = train["timestamp"].dt.hour.astype(np.int8)

train['day'] = train['timestamp'].dt.day.astype(np.int8)

train["weekday"] = train["timestamp"].dt.weekday.astype(np.int8)

train["month"] = train["timestamp"].dt.month.astype(np.int8)
print('Total rows in train: ', len(train))

print('Total buildings: ', train['building_id'].nunique(), '\t', np.sort(train['building_id'].unique()))

print('Number of meter types: ', train['meter'].nunique(), '\t', np.sort(train['meter'].unique()))

print('Total hourly intervals: ', train['timestamp'].nunique(), '\t', np.sort(train['timestamp'].unique()))
print('Count of ', train.groupby('meter')['building_id'].nunique())

print('\n', 'Total meters: ', train.groupby('meter')['building_id'].nunique().sum())
train['target'] = np.log1p(train['meter_reading'])
meter_timestamp = {'h':[], 'd':[], 'w':[], 'm':[]}



for i in range(4):

    x = train.query('meter == @i')

    meter_timestamp['h'].append(x.groupby('hour')['target'].mean())

    meter_timestamp['d'].append(x.groupby('day')['target'].mean())

    meter_timestamp['w'].append(x.groupby('weekday')['target'].mean())

    meter_timestamp['m'].append(x.groupby('month')['target'].mean())
plt.figure(figsize=(12,12))

plt.suptitle('Variation of readings for meter types with various time intervals', fontsize=16)

pos = 0

for key,value in meter_timestamp.items():

    plt.subplot(2, 2, pos+1)

    for meter,readings in enumerate(value):

        readings.plot(label=meter)

    pos += 1

    plt.legend()

plt.show()
plt.figure(figsize=(12, 12))

plt.suptitle('Distribution of meter readings by meter type', fontsize=16)

labels = {0:'Electricity', 1:'Chilled Water', 2:'Steam', 3:'Hot Water'}

colors = {0:'navy', 1:'darkorange', 2:'green', 3:'maroon'}

for i in range(4):

    plt.subplot(2,2,i+1)

    meter = train[train['meter']==i]['target']

    g = sns.distplot(meter, color=colors[i])

    plt.xlabel(labels[i])

plt.show()
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



electricity = train[train['meter']==0]

plt.figure(figsize=(20, 40))

for i in range(200):

    plt.subplot(20, 10, i+1)

    bldg = electricity[electricity['building_id']==i]

    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[0])

    plt.title("building %d" %i,fontsize=10)

    plt.axis('off')

    plt.ylim(0,2000)

plt.show()
chilledwater = train[train['meter']==1]

bldg_cw = chilledwater['building_id'].unique()



plt.figure(figsize=(20, 20))

for i in range(100):

    index = bldg_cw[i]

    bldg = chilledwater[chilledwater['building_id']==index]

    plt.subplot(10, 10, i+1)

    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[1])

    plt.title("building %d" %index,fontsize=10)

    plt.axis('off')

    plt.ylim(0, 6000)

plt.show()
steam = train[train['meter']==2]

bldg_st = steam['building_id'].unique()



plt.figure(figsize=(20, 20))

for i in range(100):

    index = bldg_st[i]

    bldg = steam[steam['building_id']==index]

    plt.subplot(10, 10, i+1)

    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[2])

    plt.title("building %d" %index,fontsize=10)

    plt.axis('off')

    plt.ylim(0, 10000)

plt.show()
hotwater = train[train['meter']==3]

bldg_hw = hotwater['building_id'].unique()



plt.figure(figsize=(20, 20))

for i in range(100):

    index = bldg_hw[i]

    bldg = hotwater[hotwater['building_id']==index]

    plt.subplot(10, 10, i+1)

    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[3])

    plt.title("building %d" %index,fontsize=10)

    plt.axis('off')

    plt.ylim(0, 4000)

plt.show()
building.head()
building.describe()
print('Total rows in building metadata: ', len(building))

print('Total sites: ', building['site_id'].nunique(), '\t', np.sort(building['site_id'].unique()))

print('Total buildings: ', building['building_id'].nunique(), '\t', np.sort(building['building_id'].unique()))
building_site = building.groupby('site_id')['building_id'].count()

plt.figure(figsize=(8,6))

plt.bar(np.arange(16), building_site)

plt.title('Number of buildings by Site', fontsize=16)

plt.xlabel('site_id')

plt.ylabel('Number')

plt.show()
plt.figure(figsize=(12,6))

g = sns.countplot(y='primary_use',data=building)

plt.title('Count of primary uses of the buildings', fontsize=16)

plt.yticks(fontsize=10)

plt.show()
plt.figure(figsize=(12, 6))

g = sns.distplot(building['year_built'].dropna(),bins=24,kde=False)

plt.title('Distribution of buildings by the year they are built', fontsize=16)

plt.yticks(fontsize=10)

plt.show()
plt.figure(figsize=(12, 7))

g = sns.scatterplot(y='square_feet',x='floor_count',hue='primary_use',size='square_feet',data=building)

plt.title('XY plot of square feet vs floor count', fontsize=16)

plt.yticks(fontsize=10)

plt.show()
weather_train.head()
# convert 'timestamp' which is a string into datetime object

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
weather_train.describe()
print('Total rows in weather_train: ', len(weather_train))

print('Total sites: ', weather_train['site_id'].nunique(), '\t', np.sort(weather_train['site_id'].unique()))

print('Total hourly intervals: ', weather_train['timestamp'].nunique(), np.sort(weather_train['timestamp'].unique()))
weather_test.head()
# convert 'timestamp' which is a string into datetime object

weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])
weather_test.describe()
print('Total rows in weather_test: ', len(weather_test))

print('Total sites: ', weather_test['site_id'].nunique(), np.sort(weather_test['site_id'].unique()))

print('Total hourly intervals: ', weather_test['timestamp'].nunique(), np.sort(weather_test['timestamp'].unique()))
# let's see if the test data correspond to the train data since this is a time series prediction challenge

test.head()
# convert 'timestamp' which is a string into datetime object

test['timestamp'] = pd.to_datetime(test['timestamp'])



# let's also extract the pertinent date features like 'hour', 'day', 'day of week' and 'month'

test["hour"] = test["timestamp"].dt.hour.astype(np.int8)

test['day'] = test['timestamp'].dt.day.astype(np.int8)

test["weekday"] = test["timestamp"].dt.weekday.astype(np.int8)

test["month"] = test["timestamp"].dt.month.astype(np.int8)
print('Total rows in test: ', len(test))

print('Total buildings: ', test['building_id'].nunique(), '\t', np.sort(test['building_id'].unique()))

print('Number of meter types: ', test['meter'].nunique(), '\t', np.sort(test['meter'].unique()))

print('Total hourly intervals: ', test['timestamp'].nunique(), '\t', np.sort(test['timestamp'].unique()))
print('Count of ', test.groupby('meter')['building_id'].nunique())

print('\n', 'Total meters: ', test.groupby('meter')['building_id'].nunique().sum())