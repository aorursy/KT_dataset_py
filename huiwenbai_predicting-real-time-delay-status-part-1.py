# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.model_selection import StratifiedKFold

from sklearn.manifold import TSNE



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = '/kaggle/input/on-time-performance/trainView.csv'

trainView_df = pd.read_csv(file_path)

trainView_df.head()
file_path = '/kaggle/input/on-time-performance/otp.csv'

otp_df = pd.read_csv(file_path)

otp_df.head()
con = sqlite3.connect('/kaggle/input/on-time-performance/database.sqlite') 

tables_df = pd.read_sql_query('SELECT name FROM sqlite_master WHERE type="table"', con)

tables_df
otp_sqlite_df = pd.read_sql_query('SELECT * FROM otp', con)

otp_sqlite_df.head()
otp_df.info()
otp_sqlite_df.info()
print(otp_df.describe())

print(otp_sqlite_df.describe())
otp_df.info()
otp_df['date'] = pd.to_datetime(otp_df['date'])

otp_df['timeStamp'] = pd.to_datetime(otp_df['timeStamp'])
otp_df['timeStamp'] = otp_df['timeStamp'].apply(lambda x: x.replace(second=0))
def status2min(x):

  x_ls = x.split(' ')

  if x == 'On Time':

    return 0

  elif x_ls[1] == 'min':

    return int(x_ls[0])

  else:

    raise ValueError



otp_df['status'] = otp_df['status'].apply(status2min)
otp_df['status_dt'] = otp_df['status'].apply(lambda x: datetime.timedelta(minutes=x))
otp_df.info()
otp_df.head()
trainView_df.info()
trainView_df['date'] = pd.to_datetime(trainView_df['date'])

trainView_df['timeStamp0'] = pd.to_datetime(trainView_df['timeStamp0'])

trainView_df['timeStamp1'] = pd.to_datetime(trainView_df['timeStamp1'])
trainView_df['timeStamp0'] = trainView_df['timeStamp0'].apply(lambda x: x.replace(second=0))

trainView_df['timeStamp1'] = trainView_df['timeStamp1'].apply(lambda x: x.replace(second=0))
def status2int(x):

  if x == 'None':

    return 0

  else:

    return int(x)



trainView_df['status'] = trainView_df['status'].apply(status2int)
trainView_df['status_dt'] = trainView_df['status'].apply(lambda x: datetime.timedelta(minutes=x))
trainView_df.info()
trainView_df.head()
test_df = otp_df[otp_df['train_id'] == '598'].copy()

test_df = test_df.groupby('date')['train_id'].count().reset_index()



fig, ax = plt.subplots(figsize=[10,5])

ax.bar(test_df['date'], test_df['train_id'])

ax.set_xlabel('Date')

ax.set_ylabel('Show-up times')

ax.set_title('How many times the train id 598 showed up in each day?')

plt.show()
test_df = otp_df[(otp_df['date'] == '2016-03-24') & (otp_df['train_id'] == '598')].copy()

test_df['hour'] = test_df['timeStamp'].dt.hour

test_df = test_df.groupby('hour')['train_id'].count().reset_index()



fig, ax = plt.subplots(figsize=[10,5])

ax.bar(test_df['hour'], test_df['train_id'])

ax.set_xlabel('Hour')

ax.set_ylabel('Show-up times')

ax.set_title('How many times the train id 598 showed up in each hour?')

plt.show()
test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])

test_df = test_df.groupby('date')['train_id'].count()



fig, ax = plt.subplots(figsize=[10,5])

ax.plot(test_df)

ax.set_xlabel('Date')

ax.set_ylabel('Num of trains')

ax.set_title('How many trains are running every day?')

plt.show()
test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])

test_df = test_df.groupby('date')['train_id'].count().reset_index()

test_df['month'] = test_df['date'].dt.month

test_df = test_df[['month', 'train_id']].groupby('month').agg(np.mean).reset_index()



fig, ax = plt.subplots(figsize=[7,5])

ax.bar(test_df['month'], test_df['train_id'])

ax.set_xlabel('Month')

ax.set_ylabel('Number of trains')

ax.set_title('Average train numbers running every day in each month')

plt.show()
test_df = trainView_df.copy().drop_duplicates(subset=['train_id', 'date'])

test_df = test_df.groupby('date')['train_id'].count().reset_index()

test_df['DOW'] = test_df['date'].dt.dayofweek + 1

test_df = test_df.groupby('DOW')['train_id'].agg(np.mean).reset_index()

test_df



fig, ax = plt.subplots(figsize=[7,5])

sns.barplot(x='DOW', y='train_id', data=test_df, palette="Blues_d")

ax.set_xlabel('Day of Week')

ax.set_ylabel('Average number of trains')

ax.set_title('Average train numbers running each day of week')

plt.show()
test_df = trainView_df.copy()

test_df['hour'] = test_df['timeStamp0'].dt.hour

test_df.drop_duplicates(subset=['train_id', 'date', 'hour'], inplace=True)

test_df = test_df.groupby(['date','hour'])['train_id'].count().reset_index()

test_df = test_df.groupby('hour').agg(np.mean).reset_index()



fig, ax = plt.subplots(figsize=[10,5])

sns.barplot(x='hour', y='train_id', data=test_df, palette="Blues_d")

ax.set_xlabel('Hour')

ax.set_ylabel('Average number of trains')

ax.set_title('Average train numbers running each hour')

plt.show()
test_df = trainView_df.copy()

test_df['service'] = test_df['service'].str.upper()

test_df = test_df.groupby('service')['train_id'].count()

test_df = test_df.sort_values(ascending=False).head(20).reset_index()



fig, ax = plt.subplots(figsize=[5,8])

sns.barplot(y='service', x='train_id', data=test_df, palette="Blues_d")

ax.set_xlabel('Count (log scale)')

ax.set_ylabel('Service type')

ax.set_title('Top 10 frequent service type')

ax.set_xscale('log')

plt.show()



# release the RAM storing test_df

test_df = []
columns_to_drop = ['lon', 'lat', 'track_change', 'track',

                   'service', 'timeStamp1', 'seconds']



trainView_df = trainView_df.drop(columns=columns_to_drop)
trainView_df = trainView_df[trainView_df['status'] < 999]
trainView_df['status'].describe()
fig, ax = plt.subplots(figsize=[10,5])

sns.distplot(trainView_df['status'])

ax.set_xlabel('Delay time (min)')

ax.set_ylabel('Density')

ax.set_title('Distribution of delay time')

plt.show()
otp_df = otp_df[otp_df['next_station'] != 'None']

otp_df = otp_df[otp_df['status'] < 999]
schedule_df = otp_df.copy()



change_col_name = {'next_station': 'arrival_station'}

schedule_df = schedule_df.rename(columns=change_col_name)



schedule_df['arrival_time'] = schedule_df['timeStamp'] - schedule_df['status_dt']

schedule_df['arrival_hour'] = schedule_df['arrival_time'].dt.hour

schedule_df['arrival_date'] = schedule_df['arrival_time'].apply(lambda x: x.date())



col_to_keep = ['train_id', 'direction', 'origin', 'arrival_station', 'date',

               'arrival_date', 'arrival_hour']

schedule_df = schedule_df[col_to_keep]



schedule_df.head()
schedule_df[schedule_df['train_id'] == '778'].head(10)
otp_df['hour'] = otp_df['timeStamp'].dt.hour
trainView_df['hour'] = trainView_df['timeStamp0'].dt.hour
otp_direction_label_df = otp_df.copy()

col_to_keep = ['train_id', 'direction', 'next_station', 'date', 'hour']

otp_direction_label_df = otp_direction_label_df[col_to_keep]

otp_direction_label_df = otp_direction_label_df.drop_duplicates()

otp_direction_label_df.head()
trainView_dir_df = trainView_df.merge(otp_direction_label_df, indicator=True, 

                                      left_on=['train_id', 'next_station', 'date', 'hour'],

                                      right_on=['train_id', 'next_station', 'date', 'hour'])

trainView_dir_df.head()
print(trainView_df.shape)

print(trainView_dir_df.shape)
col_to_keep = ['train_id', 'direction', 'status', 'next_station', 

               'date', 'timeStamp0', 'hour']

trainView_dir_df = trainView_dir_df[col_to_keep]

trainView_dir_df = trainView_dir_df.drop_duplicates()

print(trainView_dir_df.shape)

trainView_dir_df.head()
change_col_name = {'timeStamp0': 'timeStamp'}

trainView_dir_df = trainView_dir_df.rename(columns=change_col_name)

trainView_dir_df.head()
otp_combine_df = otp_df.copy()

col_to_keep = ['train_id', 'direction', 'status', 'next_station', 

               'date', 'timeStamp', 'hour']

otp_combine_df = otp_combine_df[col_to_keep]

otp_combine_df = otp_combine_df.drop_duplicates()

otp_combine_df.head()
combine_df = pd.concat([trainView_dir_df, otp_combine_df]).sort_values('timeStamp')

print(combine_df.shape)

combine_df = combine_df.drop_duplicates()

print(combine_df.shape)

combine_df.head()
station_hour_count_df = schedule_df[['arrival_station', 'direction', 'arrival_date', 'arrival_hour', 'train_id']].copy()

station_hour_count_df = station_hour_count_df.drop_duplicates()

station_hour_count_df = station_hour_count_df.groupby(['arrival_station', 'direction', 'arrival_date', 'arrival_hour']).count().reset_index()

station_hour_count_df = station_hour_count_df.rename(columns={'train_id': 'num_train'})

station_hour_count_df.head()
station_day_count_df = schedule_df[['arrival_station', 'direction', 'arrival_date', 'train_id']].copy()

station_day_count_df = station_day_count_df.drop_duplicates()

station_day_count_df = station_day_count_df.groupby(['arrival_station', 'direction', 'arrival_date']).count().reset_index()

station_day_count_df = station_day_count_df.rename(columns={'train_id': 'num_train'})

station_day_count_df.head()
sys_day_count_df = schedule_df[['direction', 'arrival_date', 'train_id']].copy()

sys_day_count_df = sys_day_count_df.drop_duplicates()

sys_day_count_df = sys_day_count_df.groupby(['arrival_date'])['train_id'].count().reset_index()

sys_day_count_df = sys_day_count_df.rename(columns={'train_id': 'num_train'})

sys_day_count_df.head()
sys_hour_count_df = schedule_df[['direction', 'arrival_date', 'train_id', 'arrival_hour']].copy()

sys_hour_count_df = sys_hour_count_df.drop_duplicates()

sys_hour_count_df = sys_hour_count_df.groupby(['arrival_date', 'arrival_hour'])['train_id'].count().reset_index()

sys_hour_count_df = sys_hour_count_df.rename(columns={'train_id': 'num_train'})

sys_hour_count_df.head()
combine_df = combine_df.sort_values('timeStamp')

combine_df = combine_df.reset_index().rename(columns={'index': 'orig_index'})

combine_df = combine_df.reset_index().rename(columns={'index': 'time_sequence'})

combine_df.head()
last_df = combine_df[['train_id', 'direction', 'time_sequence']]

last_df['last_time_sequence'] = last_df['time_sequence']

last_df = last_df.groupby(['train_id', 'direction', 'time_sequence']).sum().rolling(2).min().reset_index()

last_df = last_df[last_df['last_time_sequence'] != last_df['time_sequence']].dropna()[['time_sequence', 'last_time_sequence']]

last_df.head()
last_df = last_df.merge(combine_df, how='left', left_on='last_time_sequence',

                        right_on='time_sequence')

col_to_keep = ['time_sequence_x', 'train_id', 'direction', 'status', 

               'next_station', 'timeStamp', 'hour']

last_df = last_df[col_to_keep]

last_df.head()
change_col_name = {'time_sequence_x': 'time_sequence', 'train_id': 'last_train_id', 

                   'direction': 'last_direction', 'status': 'last_status', 

                   'next_station': 'last_next_station', 'timeStamp': 'last_timeStamp',

                   'hour':'last_hour'}

last_df = last_df.rename(columns=change_col_name)

last_df = last_df.dropna()

last_df.head()
combine_last_df = combine_df.merge(last_df, on=['time_sequence']).sort_values('timeStamp')

combine_last_df.head()
col_to_drop = ['last_train_id', 'last_direction', 'last_next_station', 'last_hour']

col = []

for i in combine_last_df.columns:

  if i not in col_to_drop:

    col.append(i)



combine_last_df = combine_last_df[col]

combine_last_df.head()
combine_last_df['delta_T'] = combine_last_df['timeStamp'] - combine_last_df['last_timeStamp']

combine_last_df.sort_values('delta_T', ascending=False).head()
fig, ax = plt.subplots(figsize=[5, 5])

sns.distplot(combine_last_df['delta_T'].dt.total_seconds())

ax.set_xlabel('Time difference (sec)')

ax.set_ylabel('Density')

ax.set_title('Distribution of time difference')

plt.show()
combine_last_df = combine_last_df[combine_last_df['delta_T'] < datetime.timedelta(minutes=90)]

combine_last_df.head()
fig, ax = plt.subplots(figsize=[5, 5])

sns.distplot(combine_last_df['delta_T'].dt.total_seconds())

ax.set_xlabel('Time difference (sec)')

ax.set_ylabel('Density')

ax.set_title('Distribution of time difference')

plt.show()
col_to_drop = ['last_timeStamp']

col = []

for i in combine_last_df.columns:

  if i not in col_to_drop:

    col.append(i)



combine_last_df = combine_last_df[col]

combine_last_df.head()
combine_last_df['delta_T_int'] = combine_last_df['delta_T'].dt.total_seconds().astype(int)/60
combine_last_df[combine_last_df['train_id'] == '778']
combine_last_df[combine_last_df['date'] == '2016-10-01']
avg_delay_df = combine_last_df.copy()

avg_delay_df = avg_delay_df.drop(columns=['time_sequence', 'delta_T', 'delta_T_int', 'last_status', 'timeStamp', 'orig_index'])

station_delay_df = avg_delay_df.groupby(['direction', 'next_station', 'date', 'hour', 'train_id'])['status'].agg(['mean']).reset_index()

station_delay_df.head()
station_delay_df = station_delay_df.groupby(['direction', 'next_station', 'date', 'hour'])['mean'].agg(['mean']).reset_index()

station_delay_df.head()
def combineTime(x):

  y = datetime.datetime.combine(x[0].date(), datetime.time(x[1],0))

  return y



station_delay_df['timeStamp'] = station_delay_df[['date', 'hour']].apply(lambda x: combineTime(x), axis=1)

station_delay_df.head()
station_delay_df = station_delay_df.rename(columns={'mean': 'avg_delay'})

station_delay_df = station_delay_df[['direction', 'next_station', 'timeStamp', 'avg_delay']]

station_delay_df.head()
sys_delay_df = avg_delay_df.groupby(['date', 'hour', 'train_id'])['status'].agg(['mean']).reset_index()

sys_delay_df.head()
sys_delay_df = sys_delay_df.groupby(['date', 'hour'])['mean'].agg(['mean']).reset_index()

sys_delay_df.head()
def combineTime(x):

  y = datetime.datetime.combine(x[0].date(), datetime.time(x[1],0))

  return y



sys_delay_df['timeStamp'] = sys_delay_df[['date', 'hour']].apply(lambda x: combineTime(x), axis=1)

sys_delay_df.head()
sys_delay_df = sys_delay_df.rename(columns={'mean': 'avg_delay'})

sys_delay_df = sys_delay_df[['timeStamp', 'avg_delay']]

sys_delay_df.head()
feature_df = combine_last_df.copy()

feature_df = feature_df.drop(columns=['time_sequence', 'orig_index', 'delta_T'])

feature_df = feature_df.rename(columns={'delta_T_int': 'delta_T'})

print(feature_df.shape)

feature_df.head()
station_day_count_df = station_day_count_df.rename(columns={'num_train': 'num_station_day',

                                                            'arrival_station': 'next_station',

                                                            'arrival_date': 'date'})

station_day_count_df['date'] = pd.to_datetime(station_day_count_df['date'])

station_day_count_df.head()
feature_df = feature_df.merge(station_day_count_df, on=['next_station', 'direction', 'date'])

print(feature_df.shape)

feature_df.head()
feature_df = feature_df.rename(columns={'num_station_day': 'num_station_day_same'})

feature_df.head()
def op_dir(x):

  if x == 'N':

    return 'S'

  else:

    return 'N'



feature_df['opp_dir'] = feature_df['direction'].apply(op_dir)

feature_df.head()
left_key = ['next_station', 'opp_dir', 'date']

right_key = ['next_station', 'direction', 'date']

feature_df = feature_df.merge(station_day_count_df, left_on=left_key, right_on=right_key)

feature_df = feature_df.drop(columns='direction_y')

feature_df = feature_df.rename(columns={'num_station_day': 'num_station_day_opp',

                                        'direction_x': 'direction'})

print(feature_df.shape)

feature_df.head()
change_col_name = {'arrival_station': 'next_station',

                   'arrival_date': 'date',

                   'arrival_hour': 'hour',

                   'num_train': 'num_station_hour'}

station_hour_count_df = station_hour_count_df.rename(columns=change_col_name)



station_hour_count_df['date'] = pd.to_datetime(station_hour_count_df['date'])



station_hour_count_df.head()
feature_df = feature_df.merge(station_hour_count_df, on=['next_station', 'direction', 'date', 'hour'])

print(feature_df.shape)

feature_df.head()
feature_df = feature_df.rename(columns={'num_station_hour': 'num_station_hour_same'})

feature_df.head()
left_key = ['next_station', 'opp_dir', 'date', 'hour']

right_key = ['next_station', 'direction', 'date', 'hour']

feature_df = feature_df.merge(station_hour_count_df, left_on=left_key, right_on=right_key)

print(feature_df.shape)

feature_df.head()
feature_df = feature_df.drop(columns='direction_y')

feature_df = feature_df.rename(columns={'direction_x': 'direction'})

feature_df = feature_df.rename(columns={'num_station_hour': 'num_station_hour_opp'})

feature_df.head()
sys_day_count_df = sys_day_count_df.rename(columns={'arrival_date': 'date',

                                                    'num_train': 'num_sys_day'})

sys_day_count_df['date'] = pd.to_datetime(sys_day_count_df['date'])

print(sys_day_count_df.shape)

print(sys_day_count_df.drop_duplicates().shape)

sys_day_count_df.head()
feature_df = feature_df.merge(sys_day_count_df, on=['date'])

print(feature_df.shape)

feature_df.head()
change_col_name = {'arrival_date': 'date',

                   'arrival_hour': 'hour',

                   'num_train': 'num_sys_hour'}

sys_hour_count_df = sys_hour_count_df.rename(columns=change_col_name)



sys_hour_count_df['date'] = pd.to_datetime(sys_hour_count_df['date'])



sys_hour_count_df.head()
feature_df = feature_df.merge(sys_hour_count_df, on=['date', 'hour'])

print(feature_df.shape)

feature_df.head()
change_col_name = {'timeStamp': 'last_hour',

                   'avg_delay': 'avg_station_same'}

station_delay_df = station_delay_df.rename(columns=change_col_name)

station_delay_df.head()
def last_hour(x):

  pure_hour = x.replace(minute=0)

  last_hour = pure_hour - datetime.timedelta(hours=1)

  return last_hour



feature_df['last_hour'] = feature_df['timeStamp'].apply(last_hour)

feature_df.head()
feature_df = feature_df.merge(station_delay_df, on=['direction', 'next_station',

                                                    'last_hour'])

print(feature_df.shape)

feature_df.head()
change_col_name = {'avg_station_same': 'avg_station_opp'}

station_delay_df = station_delay_df.rename(columns=change_col_name)

station_delay_df.head()
left_key = ['opp_dir', 'next_station', 'last_hour']

right_key = ['direction', 'next_station', 'last_hour']

feature_df = feature_df.merge(station_delay_df, left_on=left_key, right_on=right_key)

print(feature_df.shape)

feature_df.head()
feature_df = feature_df.drop(columns=['direction_y'])

feature_df = feature_df.rename(columns={'direction_x': 'direction'})

feature_df.head()
change_col_name = {'timeStamp': 'last_hour',

                   'avg_delay': 'avg_sys'}

sys_delay_df = sys_delay_df.rename(columns=change_col_name)

sys_delay_df.head()
feature_df = feature_df.merge(sys_delay_df, on='last_hour')

print(feature_df.shape)

feature_df.head()
feature_df['dow'] = feature_df['date'].dt.dayofweek + 1

feature_df['month'] = feature_df['date'].dt.month

feature_df.head()
col_to_drop = ['train_id', 'date', 'timeStamp', 'opp_dir', 'last_hour', 'next_station']

feature_df = feature_df.drop(columns=col_to_drop)

feature_df.head()
X = pd.get_dummies(feature_df, columns=['dow', 'month', 'hour', 'direction']).copy()

X = X.drop(columns='status')

print(X.shape)

X.head()
y = feature_df['status'].copy()

y.head()
X.to_csv('X.csv')

y.to_csv('y.csv')