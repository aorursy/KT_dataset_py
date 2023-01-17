import os



import matplotlib.pyplot as plt

from math import radians, cos, sin, asin, sqrt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline
FILEPATH = os.path.join("..", "input", "train.csv")

FILEPATHTEST = os.path.join("..", "input", "test.csv")

FILEPATHSAMPLE = os.path.join("..", "input", "sample_submission.csv")



train = pd.read_csv(FILEPATH)

test = pd.read_csv(FILEPATHTEST)

sample = pd.read_csv(FILEPATHSAMPLE)
test.info()

train.info()
train.head()
train.isnull().sum()
trip_duration = pd.value_counts(train["trip_duration"][train["trip_duration"] < 3600]).sort_index()

#trip_duration.plot.bar()
train.loc[train["passenger_count"] == 0].count()

train = train.loc[train['passenger_count'] > 0]
#train.loc[train["trip_duration"] < 60].count()

train = train.loc[train['trip_duration'] >= 30]
#train.loc[train["trip_duration"] > 43200].count()

train = train.loc[train['trip_duration'] <= 11000]
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;
train['weekday'] = train.loc[:,'pickup_datetime'].dt.dayofweek;

test['weekday'] = test.loc[:,'pickup_datetime'].dt.dayofweek;
pd.value_counts(test['weekday']).sort_index().plot.bar()
def calcul_distance(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    

    # Rayon (equatorial) de la terre : 6378137 m

    m = 6378137* c

    return m



def haversine_distance(x):

    x1, y1 = np.float64(x['pickup_longitude']), np.float64(x['pickup_latitude'])

    x2, y2 = np.float64(x['dropoff_longitude']), np.float64(x['dropoff_latitude'])

    return calcul_distance(x1, y1, x2, y2)
train['distance'] = train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(haversine_distance, axis=1)

test['distance'] = test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(haversine_distance, axis=1)
hour_service = pd.value_counts(train['hour']).sort_index()

hour_service.plot.bar()
features_describe = ['passenger_count', 'trip_duration', 'distance', 'hour']

train[features_describe].describe()
TARGET = ["trip_duration"]

FEATURES = ["passenger_count","vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "distance", "hour" ]

FEATURES_2 = ["weekday", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "distance", "hour" ]





y_train = train[TARGET]

X_train = train[FEATURES_2]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor(n_estimators=30,min_samples_leaf=10, min_samples_split=15, max_depth=80,verbose=0,max_features="auto",bootstrap=True,n_jobs=-1)
X_train, X_valid, y_train, y_valid = train_test_split(

    X_train, y_train, test_size=0.1, random_state=42)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring= 'neg_mean_squared_log_error')
for i in range(len(cross_val_scores)):

    cross_val_scores[i] = np.sqrt(abs(cross_val_scores[i]))

cross_val_scores
rf.fit(X_train, y_train)
loss = mean_squared_error(y_valid, rf.predict(X_valid))

loss
X_test = test[FEATURES_2]

prediction = rf.predict(X_test)

prediction
export = pd.DataFrame({'id': test.id, 'trip_duration': prediction})

export.head()

export.to_csv('submission.csv', index=False)