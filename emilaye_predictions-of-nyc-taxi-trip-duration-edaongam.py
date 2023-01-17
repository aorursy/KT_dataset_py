import os

print(os.listdir("../input"))



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



from math import radians, cos, sin, asin, sqrt

from datetime import datetime
fp1 = os.path.join("..", "input", "train.csv")

fp2 = os.path.join("..", "input", "test.csv")
train = pd.read_csv(fp1, index_col=0)

train.head() 
train.shape
train.dtypes
train.describe()
test = pd.read_csv(fp2, index_col=0)

test.head()
test.dtypes
test.shape
train.hist(bins=50, figsize=(20,15))

plt.show()
train.loc[train['trip_duration'] < 5000, 'trip_duration'].hist();



plt.title('trip_duration')

plt.show()
np.log1p(train['trip_duration']).hist();

plt.title('log_trip_duration')

plt.show()
plt.subplots(figsize=(15,5))

train.boxplot(); 
train = train[(train.trip_duration < 5000)]
train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', alpha=0.1);
train = train.loc[(train['pickup_longitude'] > -75) & (train['pickup_longitude'] < -73)]

train = train.loc[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 41)]
train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', alpha=0.1);
train = train.loc[(train['dropoff_longitude'] > -75) & (train['dropoff_longitude'] < -73)]

train = train.loc[(train['dropoff_latitude'] > 40.5) & (train['dropoff_latitude'] < 41.5)]
train['passenger_count'].hist(bins=100, log=True, figsize=(10,5));

plt.title('passenger_count')

plt.show()
train = train.loc[(train['passenger_count'] >= 0) & (train['passenger_count'] <= 6)]
train.isnull().sum()
train.duplicated().sum()
train = train.drop_duplicates()

train.duplicated().sum()
train.dtypes
train.drop(["store_and_fwd_flag"], axis=1, inplace=True)

test.drop(["store_and_fwd_flag"], axis=1, inplace=True)
train.shape, test.shape
plg, plt = 'pickup_longitude', 'pickup_latitude'

dlg, dlt = 'dropoff_longitude', 'dropoff_latitude'

pdt, ddt = 'pickup_datetime', 'dropoff_datetime'
def haversine(lon1, lat1, lon2, lat2):

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

    # Radius of earth in kilometers is 6371

    km = 6371* c

    return km



def euclidian_distance(x):

    x1, y1 = np.float64(x[plg]), np.float64(x[plt])

    x2, y2 = np.float64(x[dlg]), np.float64(x[dlt])    

    return haversine(x1, y1, x2, y2)
%time

train['distance'] = train[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)
%time

test['distance'] = test[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)
train[pdt] = train[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

train[ddt] = train[ddt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
test[pdt] = test[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

#test dataset has not "dropoff_datetiime"
train['month'] = train[pdt].apply(lambda x : x.month)

train['week_day'] = train[pdt].apply(lambda x : x.weekday())

train['day_month'] = train[pdt].apply(lambda x : x.day)

train['pickup_time_minutes'] = train[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
test['month'] = test[pdt].apply(lambda x : x.month)

test['week_day'] = test[pdt].apply(lambda x : x.weekday())

test['day_month'] = test[pdt].apply(lambda x : x.day)

test['pickup_time_minutes'] = test[pdt].apply(lambda x : x.hour * 60.0 + x.minute)
train.head()
test.head()
train.shape, test.shape
features_train = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance", "month", "week_day", "day_month", "pickup_time_minutes"]

X_train = train[features_train]

y_train = np.log1p(train["trip_duration"])



features_test = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance", "month", "week_day", "day_month", "pickup_time_minutes"]

X_test = test[features_test]
#Last check

#X_train.dtypes

#X_test.dtypes
from sklearn.ensemble import RandomForestRegressor 

#from sklearn.model_selection import GridSearchCV
#param_grid_rf = {'n_estimators' : [10, 20, 100],

                 #'min_samples_leaf' : [2, 4, 6],

                 #'max_features' : [0.2, 0.5, 'auto'],

                 #'max_depth' : [50, 80, 100]}

#rf = RandomForestRegressor()

#grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf)

#grid_search_rf.fit(X_train, y_train)
#print("Score final : ", round(grid_search_rf.score(X_train, y_train)*100, 4), " %")

#print("Meilleurs paramètres : ", grid_search_rf.best_params_)

#print("Meilleure configuration : ", grid_search_rf.best_estimator_)
#rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=15, max_depth=100, bootstrap=True, n_jobs=-1)

#rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.7, max_depth=100, bootstrap=True, n_jobs=-1)

#rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, min_samples_split=15, max_depth=100, bootstrap=True, n_jobs=-1)

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features='auto', max_depth=50, bootstrap=True, n_jobs=-1)
rf.fit(X_train, y_train)
#y_pred = grid_search_rf.predict(X_test)
log_pred = rf.predict(X_test)

y_pred = np.exp(log_pred) - np.ones(len(log_pred)) 
my_submission = pd.DataFrame({'id': test.index, 'trip_duration': y_pred})

my_submission.head()
my_submission.to_csv("submission.csv", index=False)