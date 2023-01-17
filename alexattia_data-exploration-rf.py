import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from math import radians, cos, sin, asin, sqrt 

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score

import calendar

%matplotlib inline
train = pd.DataFrame.from_csv('../input/train.csv')
def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km
train["year"] = pd.to_datetime(train['pickup_datetime']).dt.year

train["month"] = pd.to_datetime(train['pickup_datetime']).dt.month

train["day"] = pd.to_datetime(train['pickup_datetime']).dt.weekday

train["pickup_hour"] = pd.to_datetime(train['pickup_datetime']).dt.hour



# Looping through arrays of data is very slow in python. 

# Numpy provides functions that operate on entire arrays of data, 

# which lets you avoid looping and drastically improve performance

train['distance'] = haversine_np(train['pickup_longitude'],

                                 train['pickup_latitude'],

                                 train['dropoff_longitude'],

                                 train['dropoff_latitude'])

train["mean_speed"] = (train.distance / train.trip_duration)*3600

train['alone'] = (train['passenger_count']==1).apply(int)
f, ax = plt.subplots(ncols=2, nrows=3, figsize=(20,15))

train[train.distance < 30].distance.hist(bins=100, ax=ax[0,0])

ax[0, 0].axvline(train[train.distance < 30].distance.median(), color='red')

ax[0, 0].set_xlabel('Distance in km')

ax[0, 0].set_title('Traveled distance distribution')



train[train.mean_speed < 80].mean_speed.hist(bins=100, ax=ax[0,1])

ax[0, 1].axvline(train[train.mean_speed < 80].mean_speed.median(), color='red')

ax[0, 1].set_xlabel('Mean speed in km/h')

ax[0, 1].set_title('Mean speed distribution')



sns.countplot(train.month, ax =ax[1,0])

_ = ax[1,0].set_xticklabels([calendar.month_abbr[int(k.get_text())] for k in ax[1,0].get_xticklabels()])

ax[1, 0].set_title('Travel month distribution')



sns.countplot(train.day, ax =ax[1,1])

_ = ax[1,1].set_xticklabels([calendar.day_abbr[int(k.get_text())] for k in ax[1,1].get_xticklabels()])

ax[1, 1].set_title('Travel day distribution')



sns.countplot(train.pickup_hour, ax =ax[2,0])

ax[2, 0].set_title('Travel hour distribution')



train.groupby(['day', 'pickup_hour']).count()['vendor_id'].plot(ax=ax[2,1])

ax[2, 1].set_title('Travel time distribution during the week')
sns.countplot('trip_duration', data=train)

plt.yscale('log')
f, ax = plt.subplots(ncols=2, figsize=(15,5))

sns.boxplot(x='pickup_hour', y='trip_duration', data=train[train.trip_duration < 2*3600], ax = ax[0])

sns.boxplot(x='passenger_count', y='trip_duration', data=train[(train.trip_duration < 2*3600) & 

                                                               (train.passenger_count < 7)], ax = ax[1])

ax[1].set_yscale('log')

ax[0].set_yscale('log')
f, ax = plt.subplots(nrows=2, figsize=(15,10))

sns.countplot('pickup_hour', hue='alone', data=train, ax=ax[0])

sns.countplot('day', hue='alone', data=train, ax=ax[1])
_ = sns.countplot('vendor_id', data=train)
f, ax = plt.subplots(figsize=(20,5), ncols=2)

sns.countplot("passenger_count", hue='vendor_id', data=train, ax =ax[0])

_ = ax[0].set_xlim([0.5, 7])



sns.countplot("pickup_hour", hue='vendor_id', data=train, ax =ax[1])
g =sns.FacetGrid(train[train.distance < 30], hue="vendor_id", size=7)

g = g.map(sns.distplot, "distance")

g.add_legend({'green': 'vendor 1', 'blue':"vendor 2"})
for k in [0.5, 1, 5, 10, 20, 100]:

    print("{} hours+ trips : {:.4f} %".format(k, (len(train[train.trip_duration > k * 3600]) / len(train))*100))
extreme = train[train.trip_duration > 3600]

f, ax = plt.subplots(ncols=2, figsize=(15,5))

ax[0].scatter(extreme.distance, extreme.trip_duration)

ax[0].set_yscale('log')

ax[0].set_ylabel('Log Trip Duration')

ax[0].set_xlabel('Distance in km')

ax[0].set_title('Trip duration and distance for 1h+ trip')



sns.distplot(extreme['mean_speed'], ax=ax[1])

ax[1].set_ylabel('count')

ax[1].set_title('Mean speed disitriution for 1h+ trip')
print('The mean trip duration for 1h+ trip with a speed < 1 km/h is {:.2f} hour'.format(extreme[extreme.mean_speed < 1].trip_duration.mean()/3600))
df = train[train.trip_duration < 20*3600]
y = df.trip_duration

X = df.drop(['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'year', 'mean_speed'], axis=1)
le = LabelEncoder()

X.store_and_fwd_flag = le.fit_transform(X.store_and_fwd_flag)
clf = RandomForestRegressor()

clf.fit(X, y)
plt.figure(figsize=(17,5))

sns.barplot(X.columns[np.argsort(clf.feature_importances_)[::-1]], np.sort(clf.feature_importances_)[::-1])
test = pd.DataFrame.from_csv('../input/test.csv')

test["month"] = pd.to_datetime(test['pickup_datetime']).dt.month

test["day"] = pd.to_datetime(test['pickup_datetime']).dt.weekday

test["pickup_hour"] = pd.to_datetime(test['pickup_datetime']).dt.hour



test['distance'] = haversine_np(test['pickup_longitude'],

                                 test['pickup_latitude'],

                                 test['dropoff_longitude'],

                                 test['dropoff_latitude'])

test['alone'] = (test['passenger_count']==1).apply(int)

test = test.drop('pickup_datetime', axis=1)

test.store_and_fwd_flag = le.transform(test.store_and_fwd_flag)
Juputertest['trip_duration'] = clf.predict(test)