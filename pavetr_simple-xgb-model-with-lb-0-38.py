import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import datetime
from math import sin, cos, sqrt, atan2, radians, degrees
import math
import shapefile
import matplotlib.path as mplPath
import numpy as np
import json
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb
base_path1 = '../input/nyc-taxi-trip-duration/'
base_path2 = '../input/new-york-city-taxi-with-osrm/'
base_path3 = '../input/nypdcollisions/'
base_path4 = '../input/nycgeoshapes/'
base_path5 = '../input/weather-data-in-new-york-city-2016/'
train = pd.read_csv(base_path1 + 'train.csv')
test = pd.read_csv(base_path1 + 'test.csv')
### calculate distance per trip

def distance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# calculate angle between pickup and dropoff points or angle relative to NYC center

def angle_between_vectors_degrees(lat1, lon1, lat2, lon2, mode = None):
    # NYC_center
    NYC_center = [40.793209, -73.973053]
    a = np.radians(np.array([lat1, lon1]))
    b = np.radians(np.array(NYC_center))
    if mode == 'center':
        c = np.radians(np.array([NYC_center[0]+1, NYC_center[1]]))
    else:
        c = np.radians(np.array([lat2, lon2]))
    # Vectors in latitude/longitude space
    avec = a - b
    cvec = c - b

    # Adjust vectors for changed longitude scale at given latitude into 2D space
    lat = b[0]
    avec[1] *= math.cos(lat)
    cvec[1] *= math.cos(lat)
    try:
        return np.degrees(
            math.acos(np.dot(avec, cvec) / (np.linalg.norm(avec) * np.linalg.norm(cvec))))
    except ValueError:
        return 0
### get geoshapes of NYC boroughs

def get_dicty():
    json_data=open(base_path4 + 'shapes.json').read()

    data = json.loads(json_data)

    dicty = {}
    for d in data['features']:
        dicty[d['properties']['BoroName']] = [mplPath.Path([(xx[1], xx[0]) for xx in x[0]]) for x in d['geometry']['coordinates']]
    
    return dicty

def get_district(point1, point2, dicty):
    result = 'other'
    for k, v in dicty.items():
        for vv in v:
            if vv.contains_point((point1, point2)):
                result = k
                break
    return result
def process_df(df):
    dicty = get_dicty()
    
    
    df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['weekday_pickup'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['month_pickup'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['week_pickup'] = df['pickup_datetime'].apply(lambda x: x.week)
    df['hour_pickup'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['trip_distance'] = df.apply(lambda x: distance(x['pickup_latitude'], 
                                                       x['pickup_longitude'],
                                                        x['dropoff_latitude'],
                                                        x['dropoff_longitude']
                                                       ), axis = 1 )
    df['angle_start_end'] = df.apply(lambda x: angle_between_vectors_degrees(x['pickup_latitude'], x['pickup_longitude'],
                                                x['dropoff_latitude'], x['dropoff_longitude']), axis = 1)
    df['angle_direction'] = df.apply(lambda x: angle_between_vectors_degrees(x['pickup_latitude'], x['pickup_longitude'],
                                                x['dropoff_latitude'], x['dropoff_longitude'], mode = 'center'), axis = 1)
    
    df['borough_start'] = df.apply(lambda x: get_district(x['pickup_latitude'], x['pickup_longitude'],
                                                dicty), axis = 1)
    
    df['borough_end'] = df.apply(lambda x: get_district(x['dropoff_latitude'], x['dropoff_longitude'],
                                                dicty), axis = 1)
    
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y') * 1
    
    return df
train = process_df(train)
test = process_df(test)
clf = KMeans(n_clusters = 10)
clf.fit([[x] for x in train['trip_distance'].values])
train['distance_cluster'] = clf.labels_
test['distance_cluster'] = clf.predict([[x] for x in test['trip_distance'].values])
clf = KMeans(n_clusters = 20)
clf.fit([[x, y] for x, y in zip(train['pickup_latitude'].values, train['pickup_longitude'].values)])
train['pickup_coord_cluster'] = clf.labels_
test['pickup_coord_cluster'] = clf.predict([[x, y] for x, y in zip(test['pickup_latitude'].values, test['pickup_longitude'].values)])
clf = KMeans(n_clusters = 20)
clf.fit([[x, y] for x, y in zip(train['dropoff_latitude'].values, train['dropoff_longitude'].values)])
train['dropoff_coord_cluster'] = clf.labels_
test['dropoff_coord_cluster'] = clf.predict([[x, y] for x, y in zip(test['dropoff_latitude'].values, test['dropoff_longitude'].values)])
weather = pd.read_csv(base_path5 + 'weather_data_nyc_centralpark_2016.csv')
weather['date'] = weather['date'].apply(lambda x: x.replace('-', ''))
train['date'] = train['pickup_datetime'].apply(lambda x: (str(x)[8:10] if str(x)[8] != '0' else str(x)[9]) + 
                                               (str(x)[5:7] if str(x)[5] != '0' else str(x)[6]) + 
                                               str(x)[:4])
train = pd.merge(train, weather, how = 'left', on = 'date')
test['date'] = test['pickup_datetime'].apply(lambda x: (str(x)[8:10] if str(x)[8] != '0' else str(x)[9]) + 
                                               (str(x)[5:7] if str(x)[5] != '0' else str(x)[6]) + 
                                               str(x)[:4])
test = pd.merge(test, weather, how = 'left', on = 'date')

for c in ['maximum temerature', 'minimum temperature',
       'average temperature', 'precipitation', 'snow fall', 'snow depth']:
    print(c)
    mean_ = np.mean([x for x in train[c].values if type(x) != str])
    train[c] = train[c].apply(lambda x: x if type(x) != str else mean_)
    test[c] = test[c].apply(lambda x: x if type(x) != str else mean_)

train.drop('date', axis = 1, inplace = True)
test.drop('date', axis = 1, inplace = True)
routes1 = pd.read_csv(base_path2 + 'fastest_routes_train_part_1.csv')
routes2 = pd.read_csv(base_path2 + 'fastest_routes_train_part_2.csv')
routes = routes1.append(routes2, ignore_index = True)
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train = pd.merge(train, routes, how = 'left', on = 'id')


routes = pd.read_csv(base_path2 + 'fastest_routes_test.csv')
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
test = pd.merge(test, routes, how = 'left', on = 'id')


routes = pd.read_csv(base_path2 + 'second_fastest_routes_train.csv')
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
routes.columns = ['id', 'total_distance_2', 'total_travel_time_2', 'number_of_steps_2']
train = pd.merge(train, routes, how = 'left', on = 'id')

routes = pd.read_csv(base_path2 + 'second_fastest_routes_test.csv', engine = 'python',
                     delimiter = ',', error_bad_lines=False)
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
routes.columns = ['id', 'total_distance_2', 'total_travel_time_2', 'number_of_steps_2']
test = pd.merge(test, routes, how = 'left', on = 'id')

train.fillna(0, inplace = True)
test.fillna(0, inplace = True)
traffic = pd.read_csv(base_path3 + 'NYPD_Motor_Vehicle_Collisions.csv')
traffic['BOROUGH'].fillna('other', inplace = True)

train['borough_start'] = train['borough_start'].str.upper()
train['borough_end'] = train['borough_end'].str.upper()
test['borough_start'] = test['borough_start'].str.upper()
test['borough_end'] = test['borough_end'].str.upper()

traffic['time'] = traffic['TIME'].apply(lambda x: int(str(x)[:2].replace(':', '')))

### Add collisions in total by date and hour match

tr = traffic.groupby(['DATE', 'time'])['BOROUGH'].count().reset_index()
tr['DATE'] = tr['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
tr.columns = ['DATE', 'time', 'collisions_total']

train['year'] = train['pickup_datetime'].apply(lambda x: x.year)
train['month'] = train['pickup_datetime'].apply(lambda x: x.month)
train['day'] = train['pickup_datetime'].apply(lambda x: x.day)
train['time'] = train['pickup_datetime'].apply(lambda x: x.hour)

test['year'] = test['pickup_datetime'].apply(lambda x: x.year)
test['month'] = test['pickup_datetime'].apply(lambda x: x.month)
test['day'] = test['pickup_datetime'].apply(lambda x: x.day)
test['time'] = test['pickup_datetime'].apply(lambda x: x.hour)

tr['year'] = tr['DATE'].apply(lambda x: x.year)
tr['day'] = tr['DATE'].apply(lambda x: x.day)
tr['month'] = tr['DATE'].apply(lambda x: x.month)

train = pd.merge(train, tr, how = 'left', on = ['year', 'month', 'day', 'time'])
test = pd.merge(test, tr, how = 'left', on = ['year', 'month', 'day', 'time'])

train.drop(['DATE'], axis = 1, inplace = True)
test.drop(['DATE'], axis = 1, inplace = True)


### Add collisions in total by date, hour and borough match


tr = traffic.groupby(['DATE', 'time', 'BOROUGH'])['LATITUDE'].count().reset_index()
tr.columns = ['DATE', 'time', 'BOROUGH','collisions_borough_start']
tr['DATE'] = tr['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
tr['year'] = tr['DATE'].apply(lambda x: x.year)
tr['day'] = tr['DATE'].apply(lambda x: x.day)
tr['month'] = tr['DATE'].apply(lambda x: x.month)




train = pd.merge(train, tr, how = 'left', left_on = ['year', 'month', 'day', 'time', 'borough_start'],
                
                right_on = ['year', 'month', 'day', 'time', 'BOROUGH']
                )

test = pd.merge(test, tr, how = 'left', left_on = ['year', 'month', 'day', 'time', 'borough_start'],
                
                right_on = ['year', 'month', 'day', 'time', 'BOROUGH']
                )

train.drop(['BOROUGH', 'DATE'], axis = 1, inplace = True)
test.drop(['BOROUGH', 'DATE'], axis = 1, inplace = True)

tr.rename(columns = {'collisions_borough_start' : 'collisions_borough_end'}, inplace = True)

train = pd.merge(train, tr, how = 'left', left_on = ['year', 'month', 'day', 'time', 'borough_end'],
                
                right_on = ['year', 'month', 'day', 'time', 'BOROUGH']
                )

test = pd.merge(test, tr, how = 'left', left_on = ['year', 'month', 'day', 'time', 'borough_end'],
                
                right_on = ['year', 'month', 'day', 'time', 'BOROUGH']
                )

train.drop(['BOROUGH', 'DATE'], axis = 1, inplace = True)
test.drop(['BOROUGH', 'DATE'], axis = 1, inplace = True)

train.fillna(0, inplace = True)
test.fillna(0, inplace = True)
for c in test.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
        
train.drop_duplicates(subset = ['id'], keep = 'first', inplace = True)
test.drop_duplicates(subset = ['id'], keep = 'first', inplace = True)
train['borough_same'] = (train['borough_start'] == train['borough_end']) * 1
test['borough_same'] = (test['borough_start'] == test['borough_end']) * 1
train['hour_period'] = train['hour_pickup'].apply(lambda x: 0 if x <= 6 else 1 if x <= 12 else 2 if x <= 18 else 3)
test['hour_period'] = test['hour_pickup'].apply(lambda x: 0 if x <= 6 else 1 if x <= 12 else 2 if x <= 18 else 3)
train['cluster_same'] = (train['pickup_coord_cluster'] == train['dropoff_coord_cluster']) * 1
test['cluster_same'] = (test['pickup_coord_cluster'] == test['dropoff_coord_cluster']) * 1
train['lat_distance'] = train['pickup_latitude'] - train['dropoff_latitude']
test['lat_distance'] = test['pickup_latitude'] - test['dropoff_latitude']

train['lon_distance'] = train['pickup_longitude'] - train['dropoff_longitude']
test['lon_distance'] = test['pickup_longitude'] - test['dropoff_longitude']

train['week_end'] = (train['weekday_pickup'] == 0)*1 + (train['weekday_pickup'] == 6)*1
test['week_end'] = (test['weekday_pickup'] == 0)*1 + (test['weekday_pickup'] == 6)*1

full = pd.concat([train, test]).reset_index(drop=True)
coords = np.vstack((full[['pickup_latitude', 'pickup_longitude']],
                    full[['dropoff_latitude', 'dropoff_longitude']]))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + \
                             np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + \
                        np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

train['direction_ns'] = (train.pickup_latitude > train.dropoff_latitude) * 1 + 1
indices = train[(train.pickup_latitude == train.dropoff_longitude) & (train.pickup_latitude != 0)].index
train.loc[indices, 'direction_ns'] = 0

train['direction_ew'] = (train.pickup_longitude > train.dropoff_longitude) * 1 + 1
indices = train[(train.pickup_longitude == train.dropoff_longitude) & (train.pickup_longitude != 0)].index
train.loc[indices, 'direction_ew'] = 0

test['direction_ns'] = (test.pickup_latitude > test.dropoff_latitude) * 1 + 1
indices = test[(test.pickup_latitude == test.dropoff_longitude) & (test.pickup_latitude != 0)].index
test.loc[indices, 'direction_ns'] = 0

test['direction_ew'] = (test.pickup_longitude > test.dropoff_longitude) * 1 + 1
indices = test[(test.pickup_longitude == test.dropoff_longitude) & (test.pickup_longitude != 0)].index
test.loc[indices, 'direction_ew'] = 0


train['speed'] = train['trip_distance'] / train['trip_duration']
speed_cluster = train.groupby(['pickup_coord_cluster', 'hour_pickup'])['speed'].mean().reset_index()
train = pd.merge(train, speed_cluster, how = 'left', on = ['pickup_coord_cluster', 'hour_pickup'])
test = pd.merge(test, speed_cluster, how = 'left', on = ['pickup_coord_cluster', 'hour_pickup'])
train.drop('speed_x', axis = 1, inplace = True)
train.rename(columns = {'speed_y':'speed'}, inplace = True)
train.rename(columns = {'speed':'speed_pickup'}, inplace = True)
test.rename(columns = {'speed':'speed_pickup'}, inplace = True)

train['speed'] = train['trip_distance'] / train['trip_duration']
speed_cluster = train.groupby(['dropoff_coord_cluster', 'hour_pickup'])['speed'].mean().reset_index()
train = pd.merge(train, speed_cluster, how = 'left', on = ['dropoff_coord_cluster', 'hour_pickup'])
test = pd.merge(test, speed_cluster, how = 'left', on = ['dropoff_coord_cluster', 'hour_pickup'])
train.drop('speed_x', axis = 1, inplace = True)
train.rename(columns = {'speed_y':'speed'}, inplace = True)
train.rename(columns = {'speed':'speed_dropoff'}, inplace = True)
test.rename(columns = {'speed':'speed_dropoff'}, inplace = True)
train['direction'] = (train['direction_ns'] == train['direction_ew'])*1
test['direction'] = (test['direction_ns'] == test['direction_ew'])*1
train.to_csv(base_path1 + 'train_p.csv', index = False)
test.to_csv(base_path1 + 'test_p.csv', index = False)
train_cols = [
    'borough_start',
'borough_end',
  'vendor_id',
'passenger_count',
'pickup_longitude',
'pickup_latitude',
'dropoff_longitude',
'dropoff_latitude',
'store_and_fwd_flag',
'weekday_pickup',
'month_pickup',
'week_pickup',
'hour_pickup',
'trip_distance',
'angle_start_end',
'angle_direction',
'distance_cluster',
'pickup_coord_cluster',
'dropoff_coord_cluster',
'maximum temerature',
'minimum temperature',
'average temperature',
'precipitation',
'snow fall',
'snow depth',
'total_distance',
'total_travel_time',
'number_of_steps',
'total_distance_2',
'total_travel_time_2',
'number_of_steps_2' ,
'collisions_total',
'collisions_borough_start',
'collisions_borough_end',
'borough_same',
'hour_period',
'cluster_same',
'lat_distance',
'lon_distance',
'week_end',
'speed_dropoff',
'speed_pickup',
'direction',
'pickup_pca0',
'pickup_pca1',
'dropoff_pca0',
'dropoff_pca1',
'pca_manhattan',
'direction_ns',
'direction_ew'
]
train['trip_duration'] = np.log(train['trip_duration'] + 1)
X_train, X_valid, y_train, y_valid = train_test_split( train[train_cols], train['trip_duration'], test_size=0.2, random_state=42)
xgb_pars = {'min_child_weight': 10, 'eta': 0.025, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}


d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(d_train, 
              'train'), (d_valid, 'valid')]

bst = xgb.train(xgb_pars, d_train, 10**6, watchlist, early_stopping_rounds=10, verbose_eval=5, 
               maximize = False
               )
d_test = xgb.DMatrix(test[train_cols])
ytest = bst.predict(d_test)
test['trip_duration'] = np.exp(ytest) - 1
test[['id', 'trip_duration']].drop_duplicates(subset = ['id'], keep = 'first').to_csv(base_path1 + 'pavel_xgb_submission.csv.gz', index=False, compression='gzip')

