import pandas as pd

import numpy as np

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans



t0 = dt.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)





train['store_and_fwd_flag'][train.store_and_fwd_flag.values == 'Y'] = 1

test['store_and_fwd_flag'][test.store_and_fwd_flag.values == 'Y'] = 1



train['store_and_fwd_flag'][train.store_and_fwd_flag.values == 'N'] = 0

test['store_and_fwd_flag'][test.store_and_fwd_flag.values == 'N'] = 0

'''

train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())



#this is to check if there is any data which does not follow [trip_duration = dropoff_datetime - pickup_datetime]

duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]



train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

plt.hist(train['log_trip_duration'].values, bins=100)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()



plt.plot(train.groupby('pickup_date').count()[['id']], 'o-', label='train')

plt.plot(test.groupby('pickup_date').count()[['id']], 'o-', label='test')

plt.title('Train and test period complete overlap.')

plt.legend(loc=0)

plt.ylabel('number of records')

plt.show()

'''



N = 100000



city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N],

              color='blue', s=1, label='train', alpha=0.1)

ax[1].scatter(test['pickup_longitude'].values[:N], test['pickup_latitude'].values[:N],

              color='green', s=1, label='test', alpha=0.1)

fig.suptitle('Train and test area complete overlap.')

ax[0].legend(loc=0)

ax[0].set_ylabel('latitude')

ax[0].set_xlabel('longitude')

ax[1].set_xlabel('longitude')

ax[1].legend(loc=0)

plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.show()



#pca

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

                    train[['dropoff_latitude', 'dropoff_longitude']].values,

                    test[['pickup_latitude', 'pickup_longitude']].values,

                    test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]

train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]

train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]



fig, ax = plt.subplots(ncols=2)

ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N], color='blue', s=1, alpha=0.1)

ax[1].scatter(train['pickup_pca0'].values[:N], train['pickup_pca1'].values[:N], color='green', s=1, alpha=0.1)

fig.suptitle('Pickup lat long coords and PCA transformed coords.')

ax[0].set_ylabel('latitude')

ax[0].set_xlabel('longitude')

ax[1].set_xlabel('pca0')

ax[1].set_ylabel('pca1')

ax[0].set_xlim(city_long_border)

ax[0].set_ylim(city_lat_border)

pca_borders = pca.transform([[x, y] for x in city_lat_border for y in city_long_border])

ax[1].set_xlim(pca_borders[:, 0].min(), pca_borders[:, 0].max())

ax[1].set_ylim(pca_borders[:, 1].min(), pca_borders[:, 1].max())

plt.show()



#distance

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

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])



test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])



train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2

test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2



#date time features

train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']



test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']



#speed

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

fig, ax = plt.subplots(ncols=3, sharey=True)

ax[0].plot(train.groupby('pickup_hour').mean()['avg_speed_h'], 'bo-', lw=2, alpha=0.7)

ax[1].plot(train.groupby('pickup_weekday').mean()['avg_speed_h'], 'go-', lw=2, alpha=0.7)

ax[2].plot(train.groupby('pickup_week_hour').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7)

ax[0].set_xlabel('hour')

ax[1].set_xlabel('weekday')

ax[2].set_xlabel('weekhour')

ax[0].set_ylabel('average speed')

fig.suptitle('Rush hour average traffic speed')

plt.show()



train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)



# Average speed for regions

gby_cols = ['pickup_lat_bin', 'pickup_long_bin']

coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()

coord_count = train.groupby(gby_cols).count()[['id']].reset_index()

coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

coord_stats = coord_stats[coord_stats['id'] > 100]

fig, ax = plt.subplots(ncols=1, nrows=1)

ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], color='black', s=1, alpha=0.5)

ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,

           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

plt.title('Average speed')

plt.show()



train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)

train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)

train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)

train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))

test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)

test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)

test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)

test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)

test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))



#clustering

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])



train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])

train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])

test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

t1 = dt.datetime.now()

print('Time till clustering: %i seconds' % (t1 - t0).seconds)



fig, ax = plt.subplots(ncols=1, nrows=1)

ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,

           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

plt.show()
