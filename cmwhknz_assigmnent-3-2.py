# load default Python mudules
import numpy as np
import pandas as pd

# pandas display option
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set random seed 
RSEED = 100

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')
# read data in pandas dataframe, 1,000,000 takes about 2 minutes
data =  pd.read_csv('../input/train.csv', nrows = 5_000_000, parse_dates=["pickup_datetime"])

# list first few rows
data.head()
# remove n.a. data
data = data.dropna()

# extract fare_amount between 2.5 and 200
data = data[data['fare_amount'].between(left = 2.5, right = 200)]
# read the testing dataset
test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
# minimum and maximum longitude in the test set
min(test.pickup_longitude.min(), test.dropoff_longitude.min()), \
max(test.pickup_longitude.max(), test.dropoff_longitude.max())

# minimum and maximum latitude in the test set
min(test.pickup_latitude.min(), test.dropoff_latitude.min()), \
max(test.pickup_latitude.max(), test.dropoff_latitude.max())

# set the boundary based on testing dataset
BB = (
min(test.pickup_longitude.min(), test.dropoff_longitude.min()),
max(test.pickup_longitude.max(), test.dropoff_longitude.max()),
min(test.pickup_latitude.min(), test.dropoff_latitude.min()),
max(test.pickup_latitude.max(), test.dropoff_latitude.max())
)

# remove the latitude and longtitude outside the boundry
data = data.loc[data['pickup_latitude'].between(BB[2], BB[3])]
data = data.loc[data['pickup_longitude'].between(BB[0], BB[1])]
data = data.loc[data['dropoff_latitude'].between(BB[2], BB[3])]
data = data.loc[data['dropoff_longitude'].between(BB[0], BB[1])]

# absolute difference in latitude and longitude
data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()
data['no_diff'] = ((data['abs_lat_diff'] == 0) & (data['abs_lon_diff'] == 0))

# remove the 0 distance data
data = data[data['no_diff'] == False]

# this function will return distant in km
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

# calculate the distance
data['distance'] = distance(data.pickup_latitude, data.pickup_longitude, data.dropoff_latitude, data.dropoff_longitude)

data.describe()
# limit the passenger
data = data[data['passenger_count'].between(left = 1, right = 6)]
# add the time information
data['year'] = data.pickup_datetime.apply(lambda t: t.year)
data['month'] = data.pickup_datetime.apply(lambda t: t.month)
data['weekday'] = data.pickup_datetime.apply(lambda t: t.weekday())
data['hour'] = data.pickup_datetime.apply(lambda t: t.hour)
# Traffic Density

bins_lon = 500
bins_lat = 500

delta_lon = (BB[3] - BB[2])
delta_lat = (BB[1] - BB[0])

delta_lonbase = (delta_lon / bins_lon)
delta_latbase = (delta_lat / bins_lat)

data['pickup_lonbin'] = (data['pickup_longitude'] - BB[0]) //delta_lonbase
data['pickup_latbin'] = (data['pickup_latitude'] - BB[2]) //delta_latbase
data['pickup_grib'] = np.minimum(bins_lat,(data['pickup_latitude'] - BB[2]) //delta_latbase) + np.minimum(bins_lon,(data['pickup_longitude'] - BB[0]) //delta_lonbase)*1000
data['dropoff_grib'] = np.minimum(bins_lat,(data['dropoff_latitude'] - BB[2]) //delta_latbase) + np.minimum(bins_lon,(data['dropoff_longitude'] - BB[0]) //delta_lonbase)*1000
data['pickup_grib_hour'] = data['pickup_grib']*100 + data['hour']
data['dropoff_grib_hour'] = data['dropoff_grib']*100 + data['hour']

# Calculate the density in each hout
pickup_den_data = {}
dropoff_den_data = {}
for i in range(24):
    for key,value in data[data['hour']==i]['pickup_grib'].value_counts().items():
        pickup_den_data[key*100+i] = value
    for key,value in data[data['hour']==i]['dropoff_grib'].value_counts().items():
        dropoff_den_data[key*100+i] = value

# Insert the density into the data
def pickupfreq(pickup_grib_hour):
    try:
        return pickup_den_data[pickup_grib_hour]
    except KeyError:
        return 0

def dropofffreq(dropoff_grib_hour):
    try:
        return dropoff_den_data[dropoff_grib_hour]
    except KeyError:
        return  0

data['pickup_den'] = data['pickup_grib_hour'].apply(pickupfreq)
data['dropoff_den'] = data['dropoff_grib_hour'].apply(dropofffreq)

data.head()
# Working on the testing dataset

# absolute difference in latitude and longitude
test['abs_lat_diff'] = (test['dropoff_latitude'] - test['pickup_latitude']).abs()
test['abs_lon_diff'] = (test['dropoff_longitude'] - test['pickup_longitude']).abs()
test['no_diff'] = ((test['abs_lat_diff'] == 0) & (test['abs_lon_diff'] == 0))

# calculate the distance
test['distance'] = distance(test.pickup_latitude, test.pickup_longitude, test.dropoff_latitude, test.dropoff_longitude)

# add the time information on testing dataset
test['year'] = test.pickup_datetime.apply(lambda t: t.year)
test['month'] = test.pickup_datetime.apply(lambda t: t.month)
test['weekday'] = test.pickup_datetime.apply(lambda t: t.weekday())
test['hour'] = test.pickup_datetime.apply(lambda t: t.hour)

# calculate the grib position
test['pickup_lonbin'] = (test['pickup_longitude'] - BB[0]) //delta_lonbase
test['pickup_latbin'] = (test['pickup_latitude'] - BB[2]) //delta_latbase
test['pickup_grib'] = np.minimum(bins_lat,(test['pickup_latitude'] - BB[2]) //delta_latbase) + np.minimum(bins_lon,(test['pickup_longitude'] - BB[0]) //delta_lonbase)*1000
test['dropoff_grib'] = np.minimum(bins_lat,(test['dropoff_latitude'] - BB[2]) //delta_latbase) + np.minimum(bins_lon,(test['dropoff_longitude'] - BB[0]) //delta_lonbase)*1000
test['pickup_grib_hour'] = test['pickup_grib']*100 + test['hour']
test['dropoff_grib_hour'] = test['dropoff_grib']*100 + test['hour']

# insert the traffic density
test['pickup_den'] = test['pickup_grib_hour'].apply(pickupfreq)
test['dropoff_den'] = test['dropoff_grib_hour'].apply(dropofffreq)

data.drop(columns=['key', 'pickup_datetime','no_diff','pickup_lonbin','pickup_latbin','pickup_grib','dropoff_grib','pickup_grib_hour','dropoff_grib_hour'], inplace=True)
test.drop(columns=['key', 'pickup_datetime','no_diff','pickup_lonbin','pickup_latbin','pickup_grib','dropoff_grib','pickup_grib_hour','dropoff_grib_hour'], inplace=True)
data = data.sample(n = 50000)
import xgboost as xgb
from bayes_opt import BayesianOptimization

data_label = data['fare_amount']
data_features = data.drop(['fare_amount'], axis=1)

data_features.shape, data_label.shape, test.shape
dtrain = xgb.DMatrix(data_features, label=data_label)
dtest = xgb.DMatrix(test)
params = {'colsample_bytree': 1.0,
 'eta': 0.1,
 'gamma': 0.001,
 'max_delta_step': 10.0,
 'max_depth': 12,
 'min_child_weight': 20.0,
 'subsample': 1.0}
model = xgb.train(params, dtrain, num_boost_round=250)
prediction = model.predict(dtest)
sub = pd.read_csv("../input/sample_submission.csv")
sub["fare_amount"] = prediction
sub.to_csv('submission.csv', index=False)