# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display

import matplotlib.pyplot as plt

import datetime
train = pd.read_csv("../input/train.csv")
display(train.head())

print(train.shape)

print(train.isnull().sum())
fig, ax = plt.subplots(figsize=(7,7))

ax.scatter(train.pickup_longitude, train.pickup_latitude, s=0.05, cmap='jet')

ax.scatter(train.dropoff_longitude, train.dropoff_latitude, s=0.05, cmap='jet')



plt.show();
# Datetime features

train["pickup_datetime"]=pd.to_datetime(train['pickup_datetime'])

train["dropoff_datetime"]=pd.to_datetime(train['dropoff_datetime'])



# Distance feature

train["longitude_diff"] = train.dropoff_longitude - train.pickup_longitude

train["latitude_diff"] = train.dropoff_latitude -train.pickup_latitude
display(train.shape)

display(train.head())
#duration/distance

train['duration_dist'] = train.trip_duration/100*(abs(train.longitude_diff) + abs(train.latitude_diff))

train['duration_dist'] = (train.duration_dist - train.duration_dist.mean())/train.duration_dist.std()
display(train.duration_dist.describe())
train['pickup_t'] = (train.pickup_datetime.dt.hour * 3600 + train.pickup_datetime.dt.minute * 60 + train.pickup_datetime.dt.second)/3600

train['dropoff_t'] = (train.dropoff_datetime.dt.hour * 3600 + train.dropoff_datetime.dt.minute * 60 + train.dropoff_datetime.dt.second)/3600
train.plot.scatter('pickup_t', 'dropoff_t', figsize=(7,7),c='duration_dist', cmap='jet');
# Kudos to kaggler DrGuillermo

xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]

train1 = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]

train1 = train1[(train1.dropoff_longitude> xlim[0]) & (train1.dropoff_longitude < xlim[1])]

train1 = train1[(train1.pickup_latitude> ylim[0]) & (train1.pickup_latitude < ylim[1])]

train1 = train1[(train1.dropoff_latitude> ylim[0]) & (train1.dropoff_latitude < ylim[1])]

fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(train1.pickup_longitude, train1.pickup_latitude, s=0.001,c=train1.duration_dist, cmap='jet')

ax.scatter(train1.dropoff_longitude, train1.dropoff_latitude, s=0.001,c=train1.duration_dist, cmap='jet')





plt.show();
train_all = train[ abs(train.duration_dist - train.duration_dist.mean() ) < ( 3  * train.duration_dist.std() )]
train_all.plot.scatter('pickup_t','dropoff_t',s=1, figsize = (7,7), c='duration_dist',colormap='jet');
p_time = (train_all.pickup_datetime.dt.weekday * 24 + (train_all.pickup_datetime.dt.hour * 3600 + train_all.pickup_datetime.dt.minute * 60 + train_all.pickup_datetime.dt.second)/3600)/24

d_time = (train_all.dropoff_datetime.dt.weekday * 24 + (train_all.dropoff_datetime.dt.hour * 3600 + train_all.dropoff_datetime.dt.minute * 60 + train_all.dropoff_datetime.dt.second)/3600)/24

c = train_all.duration_dist

fig, ax = plt.subplots(figsize=(7,7))

ax.scatter(p_time, d_time,s=3, c=c, cmap='jet')



plt.show();
train_all=pd.get_dummies(train_all, columns=["store_and_fwd_flag"])
plt.clf()

c = train_all.duration_dist

d = train_all.store_and_fwd_flag_Y

fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(d, c)



plt.show();
plt.clf();

x = (train_all.longitude_diff - train_all.longitude_diff.mean())/train_all.longitude_diff.std()

y = (train_all.latitude_diff - train_all.latitude_diff.mean())/train_all.latitude_diff.std()

c = train_all.duration_dist

fig, ax = plt.subplots(figsize = (7,7))

ax.scatter(x,y,s=1, c=c, cmap='jet')

plt.show();