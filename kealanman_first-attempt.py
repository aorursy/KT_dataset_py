import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from haversine import haversine
%matplotlib inline
import gc 
import os
print(os.listdir("../input"))


# Class, for use in pipelines, to select certain columns from a DataFrame and convert to a numpy array
# From A. Geron: Hands-On Machine Learning with Scikit-Learn & TensorFlow, O'Reilly, 2017
# Modified by Derek Bridge to allow for casting in the same ways as pandas.DatFrame.astype
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, dtype=None):
        self.attribute_names = attribute_names
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_selected = X[self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        return X_selected.values
    
# Class, for use in pipelines, to binarize nominal-valued features (while avoiding the dummy variable trap)
# By Derek Bridge, 2017
class FeatureBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, features_values):
        self.features_values = features_values
        self.num_features = len(features_values)
        self.labelencodings = [LabelEncoder().fit(feature_values) for feature_values in features_values]
        self.onehotencoder = OneHotEncoder(sparse=False,n_values=[len(feature_values) for feature_values in features_values])
        self.last_indexes = np.cumsum([len(feature_values) - 1 for feature_values in self.features_values])
    def fit(self, X, y=None):
        for i in range(0, self.num_features):
            X[:, i] = self.labelencodings[i].transform(X[:, i])
        return self.onehotencoder.fit(X)
    def transform(self, X, y=None):
        for i in range(0, self.num_features):
            X[:, i] = self.labelencodings[i].transform(X[:, i])
            onehotencoded = self.onehotencoder.transform(X)
        return np.delete(onehotencoded, self.last_indexes, axis=1)
    def fit_transform(self, X, y=None):
        onehotencoded = self.fit(X).transform(X)
        return np.delete(onehotencoded, self.last_indexes, axis=1)
    def get_params(self, deep=True):
        return {"features_values" : self.features_values}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

# fields
fields = ['fare_amount', 'pickup_datetime', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

# Use pandas to read our training set
df = pd.read_csv("../input/train.csv",skipinitialspace=True, usecols=fields, nrows=10000000)

df.describe()
df.info()
df['pickup_datetime'] = df['pickup_datetime'].str.replace(' UTC', '')
df['pickup_datetime'] = df['pickup_datetime'].str.replace('-', '')
df['pickup_datetime'] = df['pickup_datetime'].str.replace(':', '')

date_time = df['pickup_datetime'].str.split(' ', n=1, expand=True)
df.drop(df.columns[[1]], axis=1, inplace=True)
date_time.info()
# converting and then downcast the new columns to smaller sizes
df['passenger_count'] = pd.to_numeric(df['passenger_count'], downcast='integer')
df['fare_amount'] = pd.to_numeric(df['fare_amount'], downcast='float')
df['pickup_latitude'] = pd.to_numeric(df['pickup_latitude'], downcast='float')
df['pickup_longitude'] = pd.to_numeric(df['pickup_longitude'], downcast='float')
df['dropoff_latitude'] = pd.to_numeric(df['dropoff_latitude'], downcast='float')
df['dropoff_longitude'] = pd.to_numeric(df['dropoff_longitude'], downcast='float')

# converting and then downcasting the new columns
date_time[0] = pd.to_numeric(date_time[0], downcast='integer')
date_time[1] = pd.to_numeric(date_time[1], downcast='integer')

# adding the new columns to the original dataframe
df['date'] = date_time[0].values
df["time"] = date_time[1].values

# free up the space used by date_time
date_time = None

# checkout the new df
df.head()
def calc_hav(row):
    point1= (row['pickup_latitude'],row['pickup_longitude'])
    point2= (row['dropoff_latitude'],row['dropoff_longitude'])
    return(haversine(point1,point2))

df['distance'] = df.apply(calc_hav, axis=1)


# downcast to save memory
df['distance'] = pd.to_numeric(df['distance'], downcast='float')

# missing_dist = (df['distance'].isnull().sum())
# print('Missing distance values: ', missing_dist)

# # impute missing values for distance
# df['distance'] = df['distance'].fillna(df['distance'].mean())
# df.info()

# missing_dist = (df['distance'].isnull().sum())
# print('Missing distance values: ', missing_dist)

# remove any negative fares, zero passengers and impossible coordinates
df = df[
    (df.fare_amount > 0) & 
    (df.passenger_count > 0) &
    (df.pickup_latitude > 40.5) &
    (df.pickup_latitude < 41) &
    (df.pickup_longitude > -75) &
    (df.pickup_longitude < -73) &
    (df.dropoff_latitude > 40.5) &
    (df.dropoff_latitude < 41) &
    (df.dropoff_longitude > -75) &
    (df.dropoff_longitude < -73) &
    (df.pickup_latitude != 0) &
    (df.pickup_longitude != 0) &
    (df.dropoff_latitude != 0) &
    (df.dropoff_latitude != 0) &
    (df.distance > 0)
       ]

# coordinates should fall within these confines 

#40.507754, -74.255323 # westernmost point
#40.739021, -73.700556 # easternmost point
#40.914862, -73.909555 # northernmost point
#40.496218, -74.247699 # southernmost point

df = df.reset_index(drop=True)
df.describe()
# get the labels
y = df.pop('fare_amount').values

# create the object that splits the data
kf = KFold(n_splits= 10)
# features I want to select
numeric_features = ['passenger_count', 'date', 'time','distance', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

# create pipelines
numeric_pipeline = Pipeline([
    ("selector", DataFrameSelector(numeric_features))
])

#nominal_pipeline = Pipeline([
#   ("selector", DataFrameSelector(nominal_features)),
#   ("binarizer", FeatureBinarizer([df[feature].unique() for feature in nominal_features]))
#)

# union the pipelines
pipeline = Pipeline([
    #"union", FeatureUnion([
    ("numeric_pipeline", numeric_pipeline),
    #"nominal_pipeline", nominal_pipeline)])),
    ("estimator", SGDRegressor(max_iter=100, verbose=1, shuffle=True, early_stopping=True, alpha=0.0003, learning_rate='invscaling', penalty='l2'))])
# run the pipeline
import warnings
import time

start = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print((np.sqrt(cross_val_score(pipeline, df, y, scoring="neg_mean_squared_error", cv=kf) ** 2)).mean())
end = time.time()
print((end - start)/60, 'mins')

