# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
print('The shape of Building Metadata is :',building_metadata.shape)

building_metadata.head()
print('The shape of weather_train data is :',weather_train.shape)

weather_train.head()
print('The shape of train data is :',train.shape)

train.head()
print('The shape of weather_test data is :',weather_test.shape)

weather_test.head()
print('The shape of test data is :',test.shape)

test.head()
print('Number of Buildings :', len(building_metadata.building_id.unique()))
print('Number of Sites and number of Buildings in Each site')

print(building_metadata.site_id.unique())

print(building_metadata.site_id.value_counts().sort_index())
building_count = building_metadata.primary_use.value_counts()

print('Number of Buildings by Primary Use:')

plt.figure(figsize=(15,3))

sns.barplot(building_count.index,building_metadata.primary_use.value_counts())

plt.xticks(rotation = 90)

plt.ylabel('Number of Buildings')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## Reducing memory of the data frames

building_metadata = reduce_mem_usage(building_metadata)

train = reduce_mem_usage(train)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

test = reduce_mem_usage(test)
# First merge train and building data

train = pd.merge(train,building_metadata,how = 'left')           

print(train.shape)

train.head()



# Now Merge train_building with weather_train data

train = pd.merge(train,weather_train, on = ['site_id','timestamp'], how = 'left')

print(train.shape)

train.head()
# First merge test and building data

test = pd.merge(test,building_metadata,how = 'left')           

print(test.shape)

test.head()



# Now Merge test_building with weather_test data

test = pd.merge(test,weather_test, on = ['site_id','timestamp'], how = 'left')

print(test.shape)

test.head()
train['timestamp'] = pd.to_datetime(train.timestamp)

test['timestamp'] = pd.to_datetime(test.timestamp)
def report_missing_data(df):

    print('Total Number of rows :', len(df))

    for column in df.columns:

        print(column,':', 'Missing rows:', sum(df[column].isnull()), '|', '% Missing: {:.2f}'.format(sum(df[column].isnull())*100/len(df)),'%')

report_missing_data(train)
# Get month, day, weekday,weekday name, hour etc. from the datetime object - timestamp.

train['month'] = train.timestamp.dt.month

train['day'] = train.timestamp.dt.day

train['weekday'] = train.timestamp.dt.weekday

train['hour'] = train.timestamp.dt.hour

train['weekday_name'] = train.timestamp.dt.weekday_name



# Get meter names from codes for better understanding

meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

train['meter_name'] = train.meter.map(meter_dict)



# Calculate elapsed years for each building

train['meter_year'] = train.timestamp.dt.year 

train['elapsed_years'] = train.meter_year - train.year_built



train.head()
# Reduce the size of merged train data

train = reduce_mem_usage(train)
sns.distplot(train.meter_reading, kde = False)
# plot hist of log transformed target variable

train['log_meter_reading'] = np.log(train.meter_reading + 1)

sns.distplot(train.log_meter_reading)

plt.subplot(1,3,2)

train.groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')

plt.subplot(1,3,1)

train.groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')

plt.subplot(1,3,3)

train.groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

plt.tight_layout()
#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

pivot_df = train.groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Hourly Energy consumption by Primary use by meter', stacked = True)

plt.tight_layout()
#plt.figure(figsize=(30,10))

plt.subplot(1,3,2)

train.groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')

plt.subplot(1,3,1)

train.groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )

plt.subplot(1,3,3)

train.groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )

plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

# Plot energy consumption by meter

plt.figure(figsize = (20,10))

for i in range(4):

    plt.subplot(2,2,i+1)

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')

    #plt.title('No 1099: Energy consumption by time by meter type')

    plt.legend()

    plt.tight_layout()

plt.show()
plt.subplot(1,3,1)

train.groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')

plt.subplot(1,3,2)

train.groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')

plt.subplot(1,3,3)

train.groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')
# Plot energy consumption by site

plt.figure(figsize = (20,20))

for i in range(16):

    plt.subplot(8,2,i+1)

    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Hour')

    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Day')

    plt.legend()

    plt.tight_layout() # Add tight layout in loop to prevent overlapping text

plt.show()

sites = [6,8,9]

for site in sites:

    print('site {}'.format(site), train[train.site_id == site].groupby('meter_name')['meter_reading'].sum())
pivot_df = train.groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

plt.tight_layout()





## Show values in bar plot

#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

#for p in ax.patches:

 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#plt.tight_layout()    



train.groupby('building_id')['meter_reading'].mean().plot(figsize = (20,6), color = 'green', title = 'Mean Energy consumption in all 1449 buildings')
train.groupby('building_id')['meter_reading'].mean().sort_values(ascending = False)[:10]
train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean().plot(kind = 'bar', figsize = (6,4), color = 'green', title = 'Mean Energy consumption for Building 1099 by meter')

print('% Steam Consumption for Building 1099 out of total energy consumption is :',(train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean()[1] / train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean().sum())*100,'%')
steam_1099 = train[(train.building_id == 1099) & (train.meter == 2)]['meter_reading'].sum()

steam_others = train[(train.building_id != 1099) & (train.meter == 2)]['meter_reading'].sum()

steam_total = train[train.meter == 2]['meter_reading'].sum()

print('% of Steam Consumption for Building 1099 out of total for all buildings',100*steam_1099/steam_total)
plt.figure(figsize = (20,10))

train[train.building_id != 1099][train.site_id == 13][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site 13 - No 1099 - Energy trend', label = 'By Hour')

train[train.building_id != 1099][train.site_id == 13][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label = 'By Day')

plt.legend()
train[train.building_id != 1099].groupby('building_id')['meter_reading'].mean().plot(figsize = (20,6), color = 'green', title = 'Mean Energy consumption in all buildings except Building 1099')
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Only Steam consumption : Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Only Steam Consumption for All buildings except 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('All meters except Steam : Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('All Meters - All buildings except 1099 : Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.building_id == 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(label='By hour', alpha=0.8, title = 'Energy consumption for Building 1099').set_ylabel('Meter reading', fontsize=14);

train[train.building_id == 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label='By Day', alpha=0.8, title = 'Energy consumption for Building 1099').set_ylabel('Meter reading', fontsize=14);
print('Year Built for Building 1099 is :',building_metadata[building_metadata.building_id == 1099].year_built.values)
print('Floor count for Building 1099 is :',building_metadata[building_metadata.building_id == 1099].floor_count.values)
plt.subplot(1,3,2)

train.groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')

plt.subplot(1,3,1)

train.groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')

plt.subplot(1,3,3)

train.groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

plt.tight_layout()
plt.subplot(1,3,2)

train[train.building_id != 1099].groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')

plt.subplot(1,3,1)

train[train.building_id != 1099].groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')

plt.subplot(1,3,3)

train[train.building_id != 1099].groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

plt.tight_layout()
#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

pivot_df = train.groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Hourly Energy consumption by Primary use by meter', stacked = True)

plt.tight_layout()

#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')

pivot_df = train[train.building_id != 1099].groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'No 1099: Mean Hourly Energy consumption by Primary use by meter', stacked = True)

plt.tight_layout()

#plt.figure(figsize=(30,10))

plt.subplot(1,3,2)

train.groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')

plt.subplot(1,3,1)

train.groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )

plt.subplot(1,3,3)

train.groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )

plt.tight_layout()
#plt.figure(figsize=(30,10))

plt.subplot(1,3,2)

train[train.building_id != 1099].groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')

plt.subplot(1,3,1)

train[train.building_id != 1099].groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )

plt.subplot(1,3,3)

train[train.building_id != 1099].groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )

plt.tight_layout()

fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();

fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

# Plot energy consumption by meter

plt.figure(figsize = (20,10))

for i in range(4):

    plt.subplot(2,2,i+1)

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')

    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')

    plt.legend()

    plt.tight_layout()

plt.show()
meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

# Plot energy consumption by meter

plt.figure(figsize = (20,10))

for i in range(4):

    plt.subplot(2,2,i+1)

    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')

    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')

    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')

    #plt.title('No 1099: Energy consumption by time by meter type')

    plt.legend()

    plt.tight_layout()

plt.show()
plt.subplot(1,3,1)

train.groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')

plt.subplot(1,3,2)

train.groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')

plt.subplot(1,3,3)

train.groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')
plt.subplot(1,3,1)

train[train.building_id != 1099].groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')

plt.subplot(1,3,2)

train[train.building_id != 1099].groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')

plt.subplot(1,3,3)

train[train.building_id != 1099].groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')
# Plot energy consumption by site

plt.figure(figsize = (20,20))

for i in range(16):

    plt.subplot(8,2,i+1)

    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Hour')

    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Day')

    plt.legend()

    plt.tight_layout() # Add tight layout in loop to prevent overlapping text

plt.show()
building_metadata[building_metadata.building_id == 1099].site_id.unique()
plt.figure(figsize = (20,6))

train[(train.building_id != 1099) & (train.site_id == 13)][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site 13'.format(i), label = 'By Hour')

train[(train.building_id != 1099) & (train.site_id == 13)][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label = 'By Day')
pivot_df = train.groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

plt.tight_layout()





## Show values in bar plot

#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

#for p in ax.patches:

 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#plt.tight_layout()    



pivot_df = train[train.building_id != 1099].groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')

pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

plt.tight_layout()





## Show values in bar plot

#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

#for p in ax.patches:

 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#plt.tight_layout()    
building_metadata.head()
# Define function to report the missing data in a data frame

def report_missing_data(df):

    print('Total Number of rows :', len(df))

    for column in df.columns:

        print(column,':', 'Missing rows:', sum(df[column].isnull()), '|', '% Missing: {:.2f}'.format(sum(df[column].isnull())*100/len(df)),'%')



        # Report missing data for building data

report_missing_data(building_metadata)
plt.figure(figsize = (100,100))

g = sns.FacetGrid(building_metadata, col = 'primary_use',col_wrap = 4)

g.map(sns.distplot,'square_feet', kde = False, label = 'primary_use')

g.set_xticklabels(rotation=90) # rotate all x-axis ticks for all facet subplots

for ax in g.axes.flat:

    plt.setp(ax.get_xticklabels(), visible=True) # Show x axis ticklabels for each facet sub plot

    plt.setp(ax.get_yticklabels(), visible=True) # Show y axis ticklabels for each facet sub plot

plt.tight_layout()
plt.figure(figsize = (100,100))

g = sns.FacetGrid(building_metadata, col = 'primary_use',col_wrap = 4)

g.map(sns.distplot,'floor_count', kde = False, label = 'primary_use')

g.set_xticklabels(rotation=90) # rotate all x-axis ticks for all facet subplots

for ax in g.axes.flat:

    plt.setp(ax.get_xticklabels(), visible=True) # Show x axis ticklabels for each facet sub plot

    plt.setp(ax.get_yticklabels(), visible=True) # Show y axis ticklabels for each facet sub plot

plt.tight_layout()
plt.figure(figsize = (25,6))

train.groupby('year_built')['building_id'].nunique().plot(kind = 'bar', color = 'green', title = 'Number of Buildings built by Year', rot = 90)
# Plot energy consumption by elapsed years

train.groupby('building_id')['elapsed_years','meter_reading'].mean().sort_values('elapsed_years').plot(x = 'elapsed_years',y = 'meter_reading', figsize = (20,6))
pivot_df = train.groupby(['year_built','meter_name'])['meter_reading'].mean().reset_index()

pivot_df.head()

pivot_df = pivot_df.pivot(index='year_built', columns='meter_name', values='meter_reading')

pivot_df.plot(kind = 'bar', figsize = (20,5), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)



#pivot_df['AllMeters'] = pivot_df.sum(axis = 1)

#pivot_df.drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

plt.tight_layout()
train[['year_built', 'square_feet']].groupby('year_built').mean().plot(kind = 'bar',figsize = (20,5))
train.head()
print('col index for air_temperature is :', train.columns.get_loc('air_temperature'))

print('col index for wind_speed is :', train.columns.get_loc('wind_speed'))
train.iloc[:,9:16].info()
sns.set_style('whitegrid')

site_building = building_metadata.groupby('site_id')['building_id'].nunique().sort_values(ascending=False).plot(kind = 'bar', figsize = (10,3), color = 'skyblue')

# Understand summary stats

weather_train.describe()
# Select only weather data columns from final train set

columns = train.columns[9:16]

print(columns)



# Create a function to draw histograms for weather columns

def plot_columns(df,columns,plot_type = 'hist'):

    plt.figure(figsize=(20,6))

    for i,column in enumerate(columns):

        plt.subplot(3,3,i+1)

        df[column].plot(kind = plot_type, label = column)

        plt.legend(frameon = False)

    plt.tight_layout()    

    plt.show() 



plot_columns(train,columns)  
# Select only weather data columns from final train set

columns = train.columns[9:16]

print(columns)



# Create a function to draw histograms for weather columns

def plot_columns(df,columns,plot_type = 'hist'):

    plt.figure(figsize=(20,6))

    for i,column in enumerate(columns):

        plt.subplot(3,3,i+1)

        sns.boxplot(x = df[column])

        plt.legend(frameon = False)

    plt.tight_layout()    

    plt.show() 



plot_columns(train,columns)
np.sort(train.iloc[:,9:16].cloud_coverage.unique())
np.log(train.precip_depth_1_hr+1.001).plot(kind = 'hist') # usually we add 1 to number to be log transformed to ensure input to log is positive. here since w have -1 as min value, we need to add more than 1.
fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)

train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);

axes[0,0].legend();



train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);

axes[0,1].legend();



train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);

axes[1,0].legend();



train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);

axes[1,1].legend();



plt.tight_layout()
# Split whole time series data into 4 different periods of the year with a consistent energy consumption trend

train_jan_to_apr = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([1,2,3,4])]

train_may_to_aug = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([5,6,7,8])]

train_sep_to_oct = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([9,10])]

train_nov_to_dec = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([11,12])]
fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)

train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
# Plot a heatmap between mean daily consumption and the weather features for the period Jan to Apr



# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Jan to Apr

train_jan_to_apr_corr_day = train_jan_to_apr.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

train_jan_to_apr_corr_hour = train_jan_to_apr[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()

train_jan_to_apr_corr_month = train_jan_to_apr.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()



# Create heatmap

plt.figure(figsize=(20,6))

#mask = np.zeros_like(train_corr)

#mask[np.tril_indices_from(mask)] = True

plt.subplot(1,3,1, title = 'Jan to Apr Hourly energy consumption vs Weather') # Add titles for individual subplots

sns.heatmap(train_jan_to_apr_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.subplot(1,3,2, title = 'Jan to Apr Daily energy consumption vs Weather')

sns.heatmap(train_jan_to_apr_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

#plt.subplot(1,3,3, title = 'Jan to Apr Monthly energy consumption vs Weather')

#sns.heatmap(train_jan_to_apr_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.tight_layout()
fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)

train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);

axes[0,0].legend();



train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);

axes[0,1].legend();



train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);

axes[1,0].legend();



train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);

axes[1,1].legend();



plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)

train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for May to Aug

train_may_to_aug_corr_day = train_may_to_aug.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

train_may_to_aug_corr_hour = train_may_to_aug[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()

train_may_to_aug_corr_month = train_may_to_aug.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()



## Create heatmap - May to Aug

plt.figure(figsize=(20,6))

#mask = np.zeros_like(train_corr)

#mask[np.tril_indices_from(mask)] = True

plt.subplot(1,3,1, title = 'May to Aug Hourly energy consumption vs Weather') # Add titles for individual subplots

sns.heatmap(train_may_to_aug_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.subplot(1,3,2, title = 'May to Aug Daily energy consumption vs Weather')

sns.heatmap(train_may_to_aug_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

#plt.subplot(1,3,3, title = 'May to Aug Monthly energy consumption vs Weather')

#sns.heatmap(train_may_to_aug_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.tight_layout()
fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)

train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);

train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);

axes[0,0].legend();



train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);

axes[0,1].legend();



train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);

axes[1,0].legend();



train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);

axes[1,1].legend();



plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)

train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Sep to Oct

train_sep_to_oct_corr_day = train_sep_to_oct.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

train_sep_to_oct_corr_hour = train_sep_to_oct[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()

train_sep_to_oct_corr_month = train_sep_to_oct.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()



## Create heatmap - Sep to Oct

plt.figure(figsize=(20,6))

#mask = np.zeros_like(train_corr)

#mask[np.tril_indices_from(mask)] = True

plt.subplot(1,3,1, title = 'Sep to Oct Hourly energy consumption vs Weather') # Add titles for individual subplots

sns.heatmap(train_sep_to_oct_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.subplot(1,3,2, title = 'Sep to Oct Daily energy consumption vs Weather')

sns.heatmap(train_sep_to_oct_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

#plt.subplot(1,3,3, title = 'Sep to Oct Monthly energy consumption vs Weather')

#sns.heatmap(train_sep_to_oct_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.tight_layout()
fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)

train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);

axes[0,0].legend();



train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);

axes[0,1].legend();



train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);

axes[1,0].legend();



train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);

axes[1,1].legend();



plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)

train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);

train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);

axes.legend();
# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Nov to Dec

train_nov_to_dec_corr_day = train_nov_to_dec.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

train_nov_to_dec_corr_hour = train_nov_to_dec[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()

train_nov_to_dec_corr_month = train_nov_to_dec.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()



## Create heatmap - Nov to Dec

plt.figure(figsize=(20,6))

#mask = np.zeros_like(train_corr)

#mask[np.tril_indices_from(mask)] = True

plt.subplot(1,3,1, title = 'Nov to Dec Hourly energy consumption vs Weather') # Add titles for individual subplots

sns.heatmap(train_nov_to_dec_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.subplot(1,3,2, title = 'Nov to Dec Daily energy consumption vs Weather')

sns.heatmap(train_nov_to_dec_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

#plt.subplot(1,3,3, title = 'Nov to Dec Monthly energy consumption vs Weather')

#sns.heatmap(train_nov_to_dec_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking

plt.tight_layout()
fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)

train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);

axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);

axes[0,0].legend();



train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);

axes[0,1].legend();



train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);

axes[1,0].legend();



train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);

axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);

axes[1,1].legend();



plt.tight_layout()
def plot_air_temp_sites(df):

    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)

    for i in range(16):

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Temp in Deg C ', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);

        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);

        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);

    fig.suptitle('Air Temperature by Site across Time', y = 1) # y positions the super title

    plt.tight_layout()  

plot_air_temp_sites(train)
def plot_dew_temp_sites(df):

    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)

    for i in range(16):

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Temp in Deg C ', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);

        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);

        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);

    fig.suptitle('Dew Temperature by Site across Time', y = 1) # y positions the super title

    plt.tight_layout()  

plot_dew_temp_sites(train)
def plot_sea_pressure_sites(df):

    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)

    for i in range(16):

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Pressure', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Pressure', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Pressure', fontsize=10);

        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);

        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);

    fig.suptitle('Sea Level Pressure by Site across Time', y = 1) # y positions the super title

    plt.tight_layout()  

plot_sea_pressure_sites(train)
def plot_wind_speed_sites(df):

    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)

    for i in range(16):

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Meters per sec', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Meters per sec', fontsize=10);

        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Meters per sec', fontsize=10);

        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);

        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);

    fig.suptitle('Wind Speed by Site across Time', y = 1) # y positions the super title

    plt.tight_layout()  

plot_wind_speed_sites(train)
# Remove outlier building 1099 from train to use in all data analyses

train_no_outlier = train[train.building_id != 1099]

train_no_outlier.set_index('timestamp', inplace = True)
# Create a dataframe with meter reading, meter type, time stamp and weather parameters.

train_meter_time = train_no_outlier[['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'meter_name']].groupby('meter_name')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].resample('D').mean()

train_meter_time.reset_index(inplace = True)

train_meter_time.head()
sns.pairplot(train_meter_time, hue = 'meter_name', markers = ['o','s','D','^'], diag_kws=dict(shade=False))
train_no_outlier[['cloud_coverage', 'meter_reading','air_temperature', 'dew_temperature', 'sea_level_pressure','wind_speed']].resample('D').mean().plot(figsize = (20,15), subplots = True)
train_no_outlier[['meter_reading','hour','month', 'air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'cloud_coverage']].groupby('hour')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].mean().plot(figsize = (10,10), subplots = True, title = 'By Hour')
train_no_outlier[['meter_reading','hour','month', 'air_temperature','dew_temperature', 'sea_level_pressure','wind_speed']].groupby('month')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].mean().plot(figsize = (10,10), subplots = True, title = 'By Month')
# Create a dataframe with meter reading, meter type, time stamp and weather parameters.

train_meter_cloud = train_no_outlier[['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'cloud_coverage']].groupby('cloud_coverage')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].resample('D').mean()

train_meter_cloud.reset_index(inplace = True)

train_meter_cloud.head()
sns.pairplot(train_meter_cloud, hue = 'cloud_coverage')