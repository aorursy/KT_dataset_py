%matplotlib inline
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from geopy.distance import vincenty
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# new york city area

min_lon = -74.844,

min_lat = 40.026

max_lon = -72.221

max_lat = 41.372
initial_len = train.shape[0]
# ruling points outside area as outliers

train = train[train['pickup_longitude'].between(min_lon, max_lon)]

train = train[train['pickup_latitude'].between(min_lat, max_lat)]

train = train[train['dropoff_longitude'].between(min_lon, max_lon)]

train = train[train['dropoff_latitude'].between(min_lat, max_lat)]
cleaned_len = train.shape[0]
# outliers removed

initial_len - cleaned_len
def get_distance(row):

    p1 = (row['pickup_latitude'], row['pickup_longitude'])

    p2 = (row['dropoff_latitude'], row['dropoff_longitude'])

    return vincenty(p1, p2).meters
train['distance'] = train.apply(get_distance, axis=1)
lon = train.loc[0]['pickup_longitude']

lat = train.loc[0]['pickup_latitude']
# check for na's

train.isnull().sum()
# to datetime

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
# get day and hour information

train['pickup_hour'] = train['pickup_datetime'].apply(lambda x: x.hour)

train['pickup_day'] = train['pickup_datetime'].apply(lambda x: x.weekday())
# convert timestamps to float

train['pickup_datetime'] = train['pickup_datetime'].apply(lambda x: x.timestamp())

train['dropoff_datetime'] = train['dropoff_datetime'].apply(lambda x: x.timestamp())
# get dummy variables for categorial data

train['store_and_fwd_flag'] = pd.get_dummies(train['store_and_fwd_flag'])
train = train.drop('id', axis=1)
from scipy.stats import pearsonr
pearsonr(train['distance'], train['trip_duration'])
import keras



from keras.models import Sequential, Model

from keras.layers import Dense, Activation, LSTM, Merge

from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasRegressor



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
X = train.drop('trip_duration', axis=1)

y = train['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_norm = normalize(y.reshape(1, -1))
X_train.shape
y_norm.T.shape
def baseline():

    model = Sequential()

    model.add(Dense(48, input_dim=12, activation='relu', kernel_initializer='normal'))

    model.add(Dense(12, activation='relu'))

    model.add(Dense(1, activation='linear', kernel_initializer='normal'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
estimators = []

regressor = KerasRegressor(build_fn=baseline, epochs=5, batch_size=50, verbose=1)

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', regressor))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)

results = cross_val_score(pipeline, X, y_norm.T, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))