%matplotlib inline

import pandas as pd

from datetime import datetime

import seaborn as sns

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x) #instead of exponential, print the entire number

train.describe()
train.info()
print(train.isnull().sum())

#pretty cool!
# check for duplicate ids - nice, no duplicates

print('No of Duplicates, Trip IDs: {}'.format(len(train) - 

                                              len(train.drop_duplicates(subset='id'))))

# check latitude bounds, Latitude: -85 to +85

print('Latitude bounds: {} to {}'.format(

    max(train.pickup_latitude.min(), train.dropoff_latitude.min()),

    max(train.pickup_latitude.max(), train.dropoff_latitude.max())

))
# check longitude bounds, Longitude: -180 to +180

print('Longitude bounds: {} to {}'.format(

    max(train.pickup_longitude.min(), train.dropoff_longitude.min()),

    max(train.pickup_longitude.max(), train.dropoff_longitude.max())

))

# Get the vendor cnt

print('Vendors cnt: {}'.format(len(train.vendor_id.unique())))
# datetime range - 6 months worth data

print('Datetime range: {} to {}'.format(train.pickup_datetime.min(), 

                                        train.dropoff_datetime.max()))

# passenger count - 10, Hmm must be a big ride or big car! BTw, guess which car?

print('Passengers: {} to {}'.format(train.passenger_count.min(), 

                                        train.passenger_count.max()))
m = np.mean(train['trip_duration'])

s = np.std(train['trip_duration'])

train = train[train['trip_duration'] <= m + 2*s]

train = train[train['trip_duration'] >= m - 2*s]
train = train[train['pickup_longitude'] <= -73.75]

train = train[train['pickup_longitude'] >= -74.03]

train = train[train['pickup_latitude'] <= 40.85]

train = train[train['pickup_latitude'] >= 40.63]

train = train[train['dropoff_longitude'] <= -73.75]

train = train[train['dropoff_longitude'] >= -74.03]

train = train[train['dropoff_latitude'] <= 40.85]

train = train[train['dropoff_latitude'] >= 40.63]
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

plt.hist(train['trip_duration'].values, bins=71)

plt.xlabel('trip_duration')

plt.ylabel('number of train records')

plt.show()
longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)

latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)

plt.figure(figsize = (10,10))

plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)

plt.show()
dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 

    'July', 'August', 'September', 'October', 'November', 'December']



#month

train["monthPickUp"] = train.pickup_datetime.dt.month.astype(np.uint8)

train["monthDropOff"] = train.dropoff_datetime.dt.month.astype(np.uint8)



#day

train["dayPickUp"] = train.pickup_datetime.dt.weekday.astype(np.uint8)

train["dayDropOff"] = train.dropoff_datetime.dt.weekday.astype(np.uint8)



#hour

train["hourPickUp"] = train.pickup_datetime.dt.hour.astype(np.uint8)

train["hourDropOff"] = train.dropoff_datetime.dt.hour.astype(np.uint8)



#for test set

#month

test["monthPickUp"] = test.pickup_datetime.dt.month.astype(np.uint8)



#day

test["dayPickUp"] = test.pickup_datetime.dt.weekday.astype(np.uint8)



#hour

test["hourPickUp"] = test.pickup_datetime.dt.hour.astype(np.uint8)
test.head()
plt.figure(figsize=(12,2))

data = train.groupby("hourPickUp").aggregate({"id":"count"}).reset_index()

sns.barplot(x='hourPickUp', y='id', data=data)

plt.title('Pick-ups Hour Distribution')

plt.xlabel('Hour of Day, 0-23')

plt.ylabel('No of Trips made')
plt.figure(figsize=(12,2))

data = train.groupby("dayPickUp").aggregate({"id":"count"}).reset_index()

sns.barplot(x="dayPickUp", y="id", data = data)

plt.title('Pick-ups based on days of the week')

plt.xlabel("days of the week Sun - Sat")

plt.ylabel("No of trips made")
plt.figure(figsize=(12,2))

data = train.groupby("monthPickUp").aggregate({"id":"count"}).reset_index()

sns.barplot(x="monthPickUp", y="id", data = data)

plt.title('Pick-ups based on months')

plt.xlabel("Months")

plt.ylabel("No of trips made")
plt.figure(figsize=(12,2))

sns.heatmap(data = pd.crosstab(train.monthPickUp,

                              train.hourPickUp,

                              values=train.vendor_id, 

                              aggfunc='count',

                              normalize='index'))

plt.title('Pickup heatmap, Month vs. Day Hour')

plt.ylabel('Month') ; plt.xlabel('Day Hour, 0-23')

passengerCount = train.groupby('passenger_count')['trip_duration'].mean()

plt.subplots(1,1,figsize=(17,10))

plt.ylabel('Time in Seconds')

sns.barplot(passengerCount.index,passengerCount.values)
train.groupby('monthPickUp').size()
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)   
train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

                    train[['dropoff_latitude', 'dropoff_longitude']].values))

from sklearn.cluster import MiniBatchKMeans

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])

train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])

test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)



fig, ax = plt.subplots(ncols=1, nrows=1)

ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], s=10, lw=0,

           c=train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

plt.show()
vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')

vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')

passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')

passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')

store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')

cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')

cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')

cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')



month_train = pd.get_dummies(train['monthPickUp'], prefix='m', prefix_sep='_')

month_test = pd.get_dummies(test['monthPickUp'], prefix='m', prefix_sep='_')

dom_train = pd.get_dummies(train['dayPickUp'], prefix='dom', prefix_sep='_')

dom_test = pd.get_dummies(test['dayPickUp'], prefix='dom', prefix_sep='_')

hour_train = pd.get_dummies(train['hourPickUp'], prefix='h', prefix_sep='_')

hour_test = pd.get_dummies(test['hourPickUp'], prefix='h', prefix_sep='_')

train.head()
train.drop(['monthDropOff','dayDropOff','hourDropOff'],axis = 1)
train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','monthPickUp','monthDropOff','dayPickUp','dayDropOff','hourDropOff','hourPickUp',

                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
test.head()
Test_id = test['id']

test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','monthPickUp','dayPickUp','hourPickUp',

                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)
train = train.drop(['dropoff_datetime','trip_duration'], axis = 1)
train.shape,test.shape