##############################################################
# Constants
##############################################################

KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)
##############################################################
# Input Parameters used with 1M training data points
##############################################################

# Training data rows to read
MAX_TRAINING_SIZE = 1_000_00

# Input parameters for DBSCAN GeoSpatial Desnity based clustering
EPS_IN_KM = 0.1           ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

# Pickup/dropoff within small radius of airports geo location
RADIUS_VICINITY_AIRPORTS = 1.0

# Thershold for trip fare rate to remove those spurious trips involving exorbitant fare rate
THERSHOLD_TRIP_FARE_RATE = 50.0

# Thereshold for compressing trip distance range from 0.0-110.x to 0.0-25.0
THRESHOLD_TRIP_DISTANCE = 25.0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import timeit
from sklearn import metrics
from haversine import haversine
start_time = timeit.default_timer()

# read data in pandas dataframe
df_train =  pd.read_csv('../input/train.csv', nrows = MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"])
df_holdout =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
test_key = df_holdout['key']
df_train.drop(columns = ['key'], inplace=True)
df_holdout.drop(columns = ['key'], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed
print('Old size: %d' % len(df_train))

### Ignore -ve fare
df_train = df_train[df_train.fare_amount >=0]

### NOTE that there is no missing or NA in 'test'
### Remove rows with NA in any field
df_train = df_train.dropna(how='any', axis='rows')

### 'test': No spurious passenger_count (min is 1 & max is 6)
### 'train': passengers_count max is 208... just 11 out of 1M trips with count > 7, hence removing those trips
df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))
### NOTE that 'test' lat-long are well within NYC boundary, whereas there are few spurious 'train' datapoints outside of NYC boundary hence removing those trips

#min(df_train.pickup_longitude.min(), df_train.dropoff_longitude.min()), max(df_train.pickup_longitude.max(), df_train.dropoff_longitude.max())
#min(df_train.pickup_latitude.min(), df_train.dropoff_latitude.min()), max(df_train.pickup_latitude.max(), df_train.dropoff_latitude.max())

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            
#Times Square (40.7590° N, 73.9845° W)
#BB = (-74.2, -73.8, 40.6, 41.0)

# load image of NYC map
BB = (-74.5, -72.8, 40.5, 41.8)

print('Old size: %d' % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print('New size: %d' % len(df_train))
# IDEA.n: Ideally, shortest route distance should be used
def addPickDropDistanceFeature(df):

    df['trip_distance'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude']))
        ),
        axis='columns'
    )
    return df

def addAirportDistanceFeatures(df):

    df['pickup_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )
    
    return df

def getAirportTrips(df, airportVicinity):
    ids = (df.pickup_distance_to_jfk < airportVicinity) | (df.drop_distance_to_jfk < airportVicinity) | (df.pickup_distance_to_lgr < airportVicinity) | (df.drop_distance_to_lgr < airportVicinity) | (df.pickup_distance_to_ewr < airportVicinity) | (df.drop_distance_to_ewr < airportVicinity)
    
    return ids
start_time = timeit.default_timer()

# Add pickup-dropoff distance feature
df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed
# With 1M datapoints, trip_distance range can be compressed from 0.0-110.83 to 0.0-25.0
# which would drop just 690 & 11 training & testing datapoints, i.e., 
# worst case impact on prediction accuracy by 0.1% (11/test_size*100)

bucketsCount = 100
feat = 'trip_distance'

df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
df_holdout[feat].hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency Log")
#(len(df_train[df_train[feat] > THRESHOLD_TRIP_DISTANCE]), len(df_holdout[df_holdout[feat] > THRESHOLD_TRIP_DISTANCE]))

print('Old size: %d' % len(df_train))
df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]
print('New size: %d' % len(df_train))
def add_datetime_features(df):
    #Convert to datetime format
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    df['hour'] = df.pickup_datetime.dt.hour
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['year'] = df.pickup_datetime.dt.year
    
    return df

start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed
hour_bins = [-1, 5, 7, 10, 16, 21, 23]
bin_names = ['late_night', 'morning', 'morning_peak', 'afternoon', 'evening', 'night']
df_train['hour_type'] = pd.cut(df_train.hour, bins=hour_bins, labels=bin_names).cat.codes
df_holdout['hour_type'] = pd.cut(df_train.hour, bins=hour_bins, labels=bin_names).cat.codes
# Merging both 'train' & 'holdout' for common feature engineering afterwhich 'holdout' data will be extracted
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)
#df_nyc_taxi.info()
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed
# Number of clusters in labels, ignoring noise if present.
n_clusters_pick = len(set(labels_pick)) - (1 if -1 in labels_pick else 0)
n_clusters_pick
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed
# Number of clusters in labels, ignoring noise if present.
n_clusters_drop = len(set(labels_drop)) - (1 if -1 in labels_drop else 0)
n_clusters_drop
df_nyc_taxi['density_DBSCAN_pickup'] = labels_pick
df_nyc_taxi['density_DBSCAN_dropoff'] = labels_drop
#df_nyc_taxi['dense_DBSCAN_trips'] = ((labels_pick != -1) & (labels_drop != -1))
'''
# NOTE that our focus of DBSCAN is not to differentiate levels of clusters, i.e., set of connected clusters => hence we are plotting all clusters together
df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, 'o')
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")
'''
'''
plt.plot(df_tmp.dropoff_longitude, df_tmp.dropoff_latitude, 'o')
plt.xlabel("Dropoff Longitude")
plt.ylabel("Dropoff Latitude")
'''
df_train = df_nyc_taxi.iloc[:train_len, :]
df_holdout = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']

(len(df_train), len(df_holdout))
# Ceiling near-zero fare values to 0.2 to check fare/dist behavior
df_train.loc[df_train.trip_distance < 0.2, 'trip_distance'] = 0.2

(df_train.fare_amount / df_train.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel('trip_rate')
plt.ylabel("Log Frequency")
df_train['trip_rate'] = df_train.apply(
    (lambda row: (row.fare_amount / row.trip_distance)),
    axis='columns'
)
#len(df_train.loc[df_train.trip_rate > THERSHOLD_TRIP_FARE_RATE])
#Trying to check if not removing 50+ trip_rate improve the score!
ids = (df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))
start_time = timeit.default_timer()

# Add airport trips distance features
df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed
#df_train.info()
# Split training data into Airport & City trips

airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout['airport_bound'] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train['airport_bound'] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[-airportTripsIds]
# Compare trip_rate for Airport & City trips
pd.DataFrame(data={'Airport Trips' : df_airport_trips.trip_rate, 'City Trips' : df_city_trips.trip_rate}).describe()
# Compare trip_rate for good -vs- poorly dense trips
#pd.DataFrame(data={'Good Density Trips' : df_train.loc[df_train.dense_DBSCAN_trips == 1].trip_rate, 'LOW Density Pickups' : df_train.loc[df_train.dense_DBSCAN_trips == 0].trip_rate}).describe()
df_train = df_train.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_rate', 'hour'])
df_train.info()
y = df_train['fare_amount']
train = df_train.drop(columns=['fare_amount'])

x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)
#Cross-validation
params = {
    # Parameters that we are going to tune.
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 1, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    # Other parameters
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}

#Block of code used for hypertuning parameters. Adapt to each round of parameter tuning.
#Turn off CV in submission
CV=False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    # Define initial best params and RMSE
    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        # Update our parameters
        params['eta'] = eta

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        # Update best RMSE
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    #Print final params to use for the model
    params['silent'] = 0 #Turn on output
    print(params)
def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test,params)
#Read and preprocess test set
x_pred = df_holdout.drop(columns = ['pickup_datetime', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour'])

#Predict from test set
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
len(test_key)
#Create submission file
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission.head()