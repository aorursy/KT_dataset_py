import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import geopy as geo

from geopy.distance import vincenty as geods



%matplotlib inline 
data_root = "../input/"

train_data_path = data_root + "/train.csv"

test_data_path = data_root + "/test.csv"
train_df = pd.read_csv(train_data_path)

test_df = pd.read_csv(test_data_path)
train_df.head()
print("number of rows: ",train_df.count()[0])

print("number of cols: ",train_df.count(axis=1)[0])
np.sum(train_df.isnull())
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])
train_df['pickup_hour'] = train_df['pickup_datetime'].dt.hour

train_df['pickup_day'] = train_df['pickup_datetime'].dt.dayofweek

train_df['pickup_day_name'] = train_df['pickup_datetime'].dt.weekday_name

train_df['dropoff_day'] = train_df['dropoff_datetime'].dt.dayofweek

train_df['trip_week'] = train_df['dropoff_datetime'].dt.week

train_df['trip_month'] = train_df['dropoff_datetime'].dt.month

train_df['trip_year'] = train_df['dropoff_datetime'].dt.year
train_df['pickup_start_point'] =   train_df[['pickup_latitude','pickup_longitude']].apply(geo.Point,axis=1)



train_df['pickup_dropoff_point'] =  train_df[['dropoff_latitude','dropoff_longitude']].apply(geo.Point,axis=1)



train_df['raw_distance'] = train_df[['pickup_start_point','pickup_dropoff_point']].apply(lambda x: geods(x[0][:2],x[1][:2]).meters,axis=1)
print(train_df['trip_year'].min(),train_df['trip_year'].max())

print(train_df['trip_month'].min(),train_df['trip_month'].max())

print(train_df['pickup_hour'].min(),train_df['pickup_hour'].max())
train_df['raw_distance'].describe()
train_df['trip_duration'].describe()
sns.distplot(train_df['trip_duration'],hist=False)
sns.regplot(x="trip_duration", y="raw_distance", data=train_df,fit_reg=False)
train_df['distance_duration_ratio'] = train_df['trip_duration'] / train_df['raw_distance']
lower_bound = train_df['distance_duration_ratio'].quantile(0.02)

upper_bound = train_df['distance_duration_ratio'].quantile(0.98)
train_df = train_df[train_df['distance_duration_ratio'] >= lower_bound]

train_df = train_df[train_df['distance_duration_ratio'] <= upper_bound]
sns.regplot(x="trip_duration", y="raw_distance", data=train_df,fit_reg=False)
sns.distplot(train_df['trip_duration'])
sns.distplot(np.log(train_df['trip_duration']))
sns.distplot(train_df['raw_distance'])
sns.distplot(np.log(train_df['raw_distance']))
sns.countplot(x=train_df['trip_month'])
sns.countplot(x=train_df['passenger_count'])
sns.countplot(x=train_df['trip_week'])
sns.countplot(x=train_df['pickup_day_name'])
sns.countplot(x=train_df['pickup_hour'])
sns.countplot(x=train_df['vendor_id'])
sns.countplot(x=train_df['store_and_fwd_flag'])

print('Y count:', np.sum(train_df['store_and_fwd_flag'] == 'Y'))
sns.countplot(x='pickup_day_name',hue='pickup_hour',data=train_df,ax = ax)
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='pickup_day_name',hue='pickup_hour',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='trip_month',hue='pickup_day_name',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.boxplot(x='vendor_id',y='trip_duration',data=train_df)

plt.ylim(0, 3000)
sns.boxplot(x='vendor_id',y='raw_distance',data=train_df)

plt.ylim(0, 10000)
train_df['trip_duration_categorized'] = pd.qcut(train_df['trip_duration'],3,labels=['short','medium','long'])

train_df['trip_distance_categorized'] = pd.qcut(train_df['raw_distance'],3,labels=['short','medium','long'])
sns.boxplot(x='trip_duration_categorized',y='trip_duration',data=train_df,)

plt.ylim(0, 3000)
sns.boxplot(x='trip_duration_categorized',y='trip_duration',hue='trip_distance_categorized',data=train_df)

plt.ylim(0, 2000)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title='distance')
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='trip_month',hue='trip_duration_categorized',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='pickup_day_name',hue='trip_duration_categorized',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='pickup_day_name',hue='trip_distance_categorized',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
_,ax = plt.subplots(1,1,figsize=(10,10))

sns.countplot(x='passenger_count',hue='trip_distance_categorized',data=train_df,ax = ax)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)