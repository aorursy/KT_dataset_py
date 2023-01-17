# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# data analysis and wrangling
import pandas as pd
import numpy as np
import datetime as dt

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
combine_df = pd.concat([train_df,test_df],ignore_index=True)
# combine_df['store_and_fwd_flag'] = combine_df['store_and_fwd_flag'] == 'Y'
combine_df['pickup_datetime'] = pd.to_datetime(combine_df['pickup_datetime'])
combine_df['dropoff_datetime'] = pd.to_datetime(combine_df['dropoff_datetime'])
combine_df['pickup_date'] = combine_df['pickup_datetime'].dt.date
combine_df['center_longitude'] = (combine_df['pickup_longitude']+combine_df['pickup_longitude'])/2
combine_df['center_latitude'] = (combine_df['dropoff_latitude']+combine_df['dropoff_latitude'])/2

long_border = (-74.03, -73.75)
lat_border = (40.63, 40.85)
longitude = list(combine_df['pickup_longitude']) + list(combine_df['dropoff_longitude'])
latitude = list(combine_df['pickup_latitude']) + list(combine_df['dropoff_latitude'])
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude, '.', markersize=0.5, alpha=0.5)
plt.xlim(long_border)
plt.ylim(lat_border)
plt.show()
#Haversine distance
def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    h = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    d = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(h))
    return d

combine_df['haversine_distance'] = haversine_distance(combine_df['pickup_latitude'],combine_df['pickup_longitude'],
                                                      combine_df['dropoff_latitude'],combine_df['dropoff_longitude'])
#Manhattan distance
def manhattan_distance(lat1, lng1, lat2, lng2):
    x = haversine_distance(lat1, lng1, lat1, lng2)
    y = haversine_distance(lat1, lng1, lat2, lng1)
    return x + y

combine_df['manhattan_distance'] = manhattan_distance(combine_df['pickup_latitude'],combine_df['pickup_longitude'],
                                                      combine_df['dropoff_latitude'],combine_df['dropoff_longitude'])
#bearing of distance traveled
def bearing_traveled(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

combine_df['bearing_traveled'] = bearing_traveled(combine_df['pickup_latitude'],combine_df['pickup_longitude'],
                                                  combine_df['dropoff_latitude'],combine_df['dropoff_longitude'])
#Date & Time
combine_df['month'] = combine_df['pickup_datetime'].dt.month
combine_df['weekofYear'] = combine_df['pickup_datetime'].dt.weekofyear
combine_df['dayofMonth'] = combine_df['pickup_datetime'].dt.day
combine_df['dayofWeek'] = combine_df['pickup_datetime'].dt.dayofweek
combine_df['hour'] = combine_df['pickup_datetime'].dt.hour
#Cluster
coordinates = np.vstack((train_df[['pickup_latitude', 'pickup_longitude']].values,
                        train_df[['dropoff_latitude', 'dropoff_longitude']].values))
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coordinates)
combine_df['pickup_cluster'] = kmeans.predict(combine_df[['pickup_latitude', 'pickup_longitude']])
combine_df['dropoff_cluster'] = kmeans.predict(combine_df[['dropoff_latitude', 'dropoff_longitude']])
combine_df['center_cluster'] = kmeans.predict(combine_df[['center_latitude', 'center_longitude']])
#Speed
combine_df['pickup_lat_bin'] = np.round(combine_df['pickup_latitude'], 3)
combine_df['pickup_long_bin'] = np.round(combine_df['pickup_longitude'], 3)
combine_df['center_lat_bin'] = np.round(combine_df['center_latitude'], 3)
combine_df['center_long_bin'] = np.round(combine_df['center_longitude'], 3)
combine_df['dropoff_lat_bin'] = np.round(combine_df['dropoff_latitude'], 3)
combine_df['dropoff_long_bin'] = np.round(combine_df['dropoff_longitude'], 3)

train_df = combine_df.iloc[:1458644,:]
test_df = combine_df.iloc[1458644:,:]
train_df['avg_speed_h'] = 1000*train_df['haversine_distance']/train_df['trip_duration']
train_df['avg_speed_m'] = 1000*train_df['manhattan_distance']/train_df['trip_duration']
coord_speed = train_df.groupby(['pickup_lat_bin','pickup_long_bin'])\
                      .agg({'avg_speed_h':np.average,'id':np.size})
coord_speed = coord_speed[coord_speed['id']>=100].reset_index()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
ax.scatter(train_df.pickup_longitude.values, train_df.pickup_latitude.values, color='black', s=0.2, alpha=0.5)
ax.scatter(coord_speed.pickup_long_bin.values, coord_speed.pickup_lat_bin.values, c=coord_speed.avg_speed_h.values,
           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)
ax.set_xlim(long_border)
ax.set_ylim(lat_border)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
plt.title('Average speed')
plt.show()
train_df['log_trip_duration'] = np.log(train_df['trip_duration']+1)

for gby_col in ['pickup_date', 'hour']:
    gby = train_df.groupby(gby_col).agg({'avg_speed_h':np.average})
    gby.columns = ['%s-gby-%s' % (col, gby_col) for col in gby.columns]
    train_df = pd.merge(train_df, gby, how='left', left_on=gby_col, right_index=True)
    test_df = pd.merge(test_df, gby, how='left', left_on=gby_col, right_index=True)

for gby_cols in [['hour','center_cluster'], ['hour', 'pickup_cluster'],  ['hour', 'dropoff_cluster'],
                 ['pickup_date','hour'], ['pickup_date','hour','pickup_cluster'], 
                 ['pickup_date','hour','center_cluster'],['pickup_date','hour','dropoff_cluster'],
                 ['pickup_lat_bin','pickup_long_bin'], ['dropoff_lat_bin','dropoff_long_bin']]:
    gby = train_df.groupby(gby_cols).agg({'avg_speed_h':np.average, 'avg_speed_m':np.average, 'id':'count'})
    gby = gby[gby['id']>=100]
    gby.columns = ['%s-gby-%s' % (col, '&'.join(gby_cols)) for col in gby.columns]
    train_df = pd.merge(train_df, gby, how='left', left_on=gby_cols, right_index=True)
    test_df = pd.merge(test_df, gby, how='left', left_on=gby_cols, right_index=True)

for gby_cols in [['pickup_cluster', 'center_cluster'],['center_cluster','dropoff_cluster'],
                 ['pickup_cluster', 'dropoff_cluster']]:
    gby = train_df.groupby(gby_cols).agg({'avg_speed_h':np.average, 'log_trip_duration':np.average, 'id':'count'})
    gby = gby[gby['id']>=100]
    gby.columns = ['%s-gby-%s' % (col, '&'.join(gby_cols)) for col in gby.columns]
    train_df = pd.merge(train_df, gby, how='left', left_on=gby_cols, right_index=True)
    test_df = pd.merge(test_df, gby, how='left', left_on=gby_cols, right_index=True)

train_df.head()
#total distance & travel time & no. of steps
train1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', 
                  usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', 
                  usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                  usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
add_data = pd.concat([train1, train2, test])
train_df = train_df.merge(add_data, how='left', on='id')
test_df = test_df.merge(add_data, how='left', on='id')
# features = ['passenger_count', 'store_and_fwd_flag', 'vendor_id',
#             'month', 'weekofYear', 'dayofMonth', 'dayofWeek', 'hour', 
#             'pickup_cluster', 'dropoff_cluster']
# for feature in features:
#     df = pd.get_dummies(combine_df[feature],prefix=feature)
#     combine_df = pd.concat([combine_df,df],axis=1).drop(feature,axis=1)
drop_features = ['pickup_date','pickup_datetime','dropoff_datetime','trip_duration', 
                  'dropoff_latitude', 'dropoff_longitude', 'dropoff_lat_bin', 'dropoff_long_bin',
                  'pickup_latitude','pickup_longitude', 'pickup_lat_bin', 'pickup_long_bin',
                  'center_latitude', 'center_longitude','center_lat_bin', 'center_long_bin']
train_df = train_df.drop(drop_features+['avg_speed_h']+['avg_speed_m'],axis=1)
test_df = test_df.drop(drop_features,axis=1)
print(train_df.info())
print(test_df.info())
X_all = train_df.drop(['id','log_trip_duration'],axis=1)
Y_all = train_df["log_trip_duration"]
X_test = test_df.drop('id',axis=1)
num_test = 0.2
X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_cv, label=Y_cv)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 'max_depth': 6,
            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 30, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)
Y_test = model.predict(dtest)
Y_test = np.exp(Y_test) - 1
submission = pd.DataFrame({
    "id": test_df["id"],
    "trip_duration": Y_test
})
submission.to_csv("submission.csv", index=False)
fig,ax = plt.subplots(figsize=(10,10))
fig = xgb.plot_importance(model, height=0.6, ax=ax)