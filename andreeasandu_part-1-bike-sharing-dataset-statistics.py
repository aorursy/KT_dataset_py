import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print(train_data.columns)
print('Train data: # rows: {}, # cols: {}'.format(train_data.shape[0], train_data.shape[1]))
print('Test data: # rows: {}, # cols: {}'.format(test_data.shape[0], test_data.shape[1]))

train_data.head()
train_data.isna().sum()
train_data['datetime'] = pd.to_datetime(train_data['datetime'])

temp = pd.DatetimeIndex(train_data['datetime'])
train_data['date'] = temp.date
train_data['hour'] = temp.time
del train_data['datetime']
train_data.head()
gb_season = train_data.groupby(['season', 'hour'])
season_summary = gb_season.size().to_frame(name='season_summary')
season_summary = (season_summary.join(gb_season.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
#                                 .join(gb_season.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())

plt.figure(figsize=(10, 5))

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary.groupby('season').groups.keys():
    plt.plot(season_summary['hour'][season_summary.groupby('season').groups[i]],
            season_summary['count_mean'][season_summary.groupby('season').groups[i]])
    
plt.grid()
plt.legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each season')
season_summary1 = gb_season.size().to_frame(name='season_summary1')
season_summary1 = (season_summary1.join(gb_season.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_season.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

season_summary2 = gb_season.size().to_frame(name='season_summary2')
season_summary2 = (season_summary2.join(gb_season.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_season.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary1.groupby('season').groups.keys():
    axes[0].plot(season_summary1['hour'][season_summary1.groupby('season').groups[i]],
            season_summary1['casual_mean'][season_summary1.groupby('season').groups[i]])

axes[0].grid()
axes[0].legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary2.groupby('season').groups.keys():
    axes[1].plot(season_summary2['hour'][season_summary2.groupby('season').groups[i]],
            season_summary2['registered_mean'][season_summary2.groupby('season').groups[i]])
    
axes[1].grid()
axes[1].legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each season', multialignment='center')
gb_weather = train_data.groupby(['weather', 'hour'])
weather_summary = gb_weather.size().to_frame(name='weather_summary')
weather_summary = (weather_summary.join(gb_weather.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
                                #.join(gb_weather.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())
plt.figure(figsize=(10, 5))
for i in weather_summary.groupby('weather').groups.keys():
# 1-good weather, 2-normal, 3-bad, 4-very bad
    plt.plot(weather_summary['hour'][weather_summary.groupby('weather').groups[i]],
            weather_summary['count_mean'][weather_summary.groupby('weather').groups[i]])
plt.grid()
plt.legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each type of weather')
weather_summary1 = gb_weather.size().to_frame(name='weather_summary1')
weather_summary1 = (weather_summary1.join(gb_weather.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_weather.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

weather_summary2 = gb_weather.size().to_frame(name='weather_summary2')
weather_summary2 = (weather_summary2.join(gb_weather.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_weather.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# 1-good, 2-normal, 3-bad, 4-very bad
for i in weather_summary1.groupby('weather').groups.keys():
    axes[0].plot(weather_summary1['hour'][weather_summary1.groupby('weather').groups[i]],
            weather_summary1['casual_mean'][weather_summary1.groupby('weather').groups[i]])

axes[0].grid()
axes[0].legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

# 1-good, 2-normal, 3-bad, 4-very bad
for i in weather_summary2.groupby('weather').groups.keys():
    axes[1].plot(weather_summary2['hour'][weather_summary2.groupby('weather').groups[i]],
            weather_summary2['registered_mean'][weather_summary2.groupby('weather').groups[i]])
    
axes[1].grid()
axes[1].legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each type of weather', multialignment='center')
# find the week day based on date.  Monday=0, Sunday=6
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['weekday'] = train_data['date'].dt.weekday
train_data.head()
gb_weekday = train_data.groupby(['weekday', 'hour'])
weekday_summary = gb_weekday.size().to_frame(name='weekday_summary')
weekday_summary = (weekday_summary.join(gb_weekday.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
                                #.join(gb_weather.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())

plt.figure(figsize=(15, 7))
for i in weekday_summary.groupby('weekday').groups.keys():
    plt.plot(weekday_summary['hour'][weekday_summary.groupby('weekday').groups[i]],
            weekday_summary['count_mean'][weekday_summary.groupby('weekday').groups[i]])
plt.grid()
plt.legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each day of the week')
weekday_summary1 = gb_weekday.size().to_frame(name='weekday_summary1')
weekday_summary1 = (weekday_summary1.join(gb_weekday.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_weekday.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

weekday_summary2 = gb_weekday.size().to_frame(name='weekday_summary2')
weekday_summary2 = (weekday_summary2.join(gb_weekday.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_weekday.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

for i in weekday_summary1.groupby('weekday').groups.keys():
    axes[0].plot(weekday_summary1['hour'][weekday_summary1.groupby('weekday').groups[i]],
            weekday_summary1['casual_mean'][weekday_summary1.groupby('weekday').groups[i]])

axes[0].grid()
axes[0].legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

for i in weekday_summary2.groupby('weekday').groups.keys():
    axes[1].plot(weekday_summary2['hour'][weekday_summary2.groupby('weekday').groups[i]],
            weekday_summary2['registered_mean'][weekday_summary2.groupby('weekday').groups[i]])

axes[1].grid()
axes[1].legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each day of the week', multialignment='center')
train_data.hist(figsize=(9, 9))
_, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

sns.countplot(x='holiday', data=train_data, ax=axes[0, 0])
sns.countplot(x='workingday', data=train_data, ax=axes[0, 1])
sns.countplot(x='season', data=train_data, ax=axes[1, 0])
sns.countplot(x='weather', data=train_data, ax=axes[1, 1])
users_count = np.array([train_data['casual'].sum(), train_data['registered'].sum()])
hist, bin_edges = np.histogram(users_count)

barlist = plt.bar( hist, bin_edges[:-1], color=['orange', 'blue'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.xlabel('casual {:{width}} registered'.format('',width=35))
train_corr = train_data.corr(method="spearman")
plt.figure(figsize=(10, 10))
sns.heatmap(train_corr, annot=True)
sns.jointplot(x='humidity', y='windspeed', data=train_data, kind='scatter');
sns.jointplot(x='humidity', y='atemp', data=train_data, kind='scatter');
sns.jointplot(x='humidity', y='temp', data=train_data, kind='scatter');
sns.jointplot(x='atemp', y='temp', data=train_data, kind='scatter');
sns.jointplot(x='windspeed', y='atemp', data=train_data, kind='scatter');
sns.jointplot(x='windspeed', y='temp', data=train_data, kind='scatter');