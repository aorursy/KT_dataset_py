import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

train.tail()
import datetime

pickup = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in train['pickup_datetime']]

pickup[1]
pickup_year = pd.Series(i.year for i in pickup)

pickup_year.value_counts()
pickup_month = pd.Series(i.month for i in pickup)

pickup_month.value_counts()
train['pickup_month'] = pd.Series(i.month for i in pickup)

train['pickup_day'] = pd.Series(i.day for i in pickup)

train['pickup_hour'] = pd.Series(i.hour for i in pickup)

train['pickup_minute'] = pd.Series(i.minute for i in pickup)

train['pickup_second'] = pd.Series(i.second for i in pickup)

train.head()
dropoff = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in train['dropoff_datetime']]

train['dropoff_day'] = pd.Series(i.day for i in dropoff)

train['dropoff_hour'] = pd.Series(i.hour for i in dropoff)

train['dropoff_minute'] = pd.Series(i.minute for i in dropoff)

train['dropoff_second'] = pd.Series(i.second for i in dropoff)

train.head()
train['store_and_fwd_flag_int'] = pd.Series(int(i=='Y') for i in train['store_and_fwd_flag'])
del train['pickup_datetime']

del train['dropoff_datetime']

del train['store_and_fwd_flag']

train.head()
plt.scatter(train['passenger_count'],train['trip_duration'])
abnormal = train[train['trip_duration'] > 1500000]

abnormal
train = train[train['trip_duration'] < 1500000]

train = train[train['passenger_count'] > 0]
plt.scatter(train['passenger_count'],train['trip_duration'])
abnormal2 = train[train['passenger_count'] > 6]

abnormal2
plt.scatter(train['pickup_longitude'],train['trip_duration'])
abnormal3 = train[train['pickup_longitude'] < -100]

abnormal3
fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(231)

plt.scatter(train['pickup_month'],train['trip_duration'])

ax2 = fig.add_subplot(232)

plt.scatter(train['pickup_day'],train['trip_duration'])

ax3 = fig.add_subplot(233)

plt.scatter(train['pickup_hour'],train['trip_duration'])

ax4 = fig.add_subplot(234)

plt.scatter(train['pickup_minute'],train['trip_duration'])

ax5 = fig.add_subplot(235)

plt.scatter(train['pickup_second'],train['trip_duration'])