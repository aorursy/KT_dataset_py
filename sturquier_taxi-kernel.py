import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from os import path

from haversine import haversine

import warnings



warnings.simplefilter("ignore")

sns.set(color_codes=True)

%matplotlib inline
INPUT_PATH = path.join('..', 'input')

train_filepath = path.join(INPUT_PATH, 'train.csv')



df = pd.read_csv(train_filepath, index_col='id')

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df.head()
test_filepath = path.join(INPUT_PATH, 'test.csv')



df_test = pd.read_csv(test_filepath, index_col='id')

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test.head()
submission_filepath = path.join(INPUT_PATH, 'sample_submission.csv')



df_submission = pd.read_csv(submission_filepath)

df_submission.head()
# df.info()

# df.describe()

# df.shape

df.head()
# df['id'].describe() 

# df['id'].nunique() # unique values



# Nothing particular underneath.

# Except the trip_duration min & max

# And number of passengers min & max



# df['vendor_id'].describe()

# df['pickup_datetime'].describe()

# df['dropoff_datetime'].describe()

# df['passenger_count'].describe()

# df['pickup_longitude'].describe()

# df['pickup_latitude'].describe()

# df['dropoff_longitude'].describe()

# df['dropoff_latitude'].describe()

# df['store_and_fwd_flag'].describe()

# df['trip_duration'].describe()
plt.figure(figsize=(20, 2))

sns.countplot(x='passenger_count', data=df);
plt.figure(figsize=(20, 2))

passengers_number_boxplot = sns.boxplot(x=df['passenger_count'])
plt.figure(figsize=(20, 5))

df['trip_duration'].hist();
df.loc[df['trip_duration'] > 7200, 'trip_duration'].sum()
plt.figure(figsize=(20, 5))

df.loc[df['trip_duration'] > 7200, 'trip_duration'].hist(bins=100);
df.loc[df['trip_duration'] < 60, 'trip_duration'].sum()
plt.figure(figsize=(20, 5))

df.loc[df['trip_duration'] < 60, 'trip_duration'].hist(bins=100);
df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude');
df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude');
# haversine() calculates the distance between 2 points on earth (kms per default)

def haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    pickup = (pickup_lat, pickup_lon)

    dropoff = (dropoff_lat, dropoff_lon)

    

    return haversine(pickup, dropoff)
def preprocess(df):

    df['store_and_fwd_flag_codes'] = df['store_and_fwd_flag'].astype('category').cat.codes

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_second'] = df['pickup_datetime'].dt.second

    df['pickup_dropoff_distance'] = df.apply(lambda x: haversine_distance(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)

    

preprocess(df)

preprocess(df_test)
# Filters we have seen previously when analysing data



trip_duration_filter = (df['trip_duration'] > 60) & (df['trip_duration'] < 7200)

passenger_count_filter = (df['passenger_count'] > 0)

pickup_filter = (df['pickup_latitude'] > 37.5) & (df['pickup_latitude'] < 45) & (df['pickup_longitude'] < -70) & (df['pickup_longitude'] > -80)

dropoff_filter = (df['dropoff_latitude'] > 38) & (df['dropoff_latitude'] < 44) & (df['dropoff_longitude'] < -70) & (df['dropoff_longitude'] > -80)



df = df[trip_duration_filter]

df = df[passenger_count_filter]

df = df[pickup_filter]

df = df[dropoff_filter]



df.shape
FEATURES = [

    'vendor_id',

    'pickup_year',

    'pickup_month',

    'pickup_day',

    'pickup_dayofweek',

    'pickup_weekday',

    'pickup_hour',

    'pickup_minute',

    'pickup_second',

    'passenger_count',

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude',

    'store_and_fwd_flag_codes',

    'pickup_dropoff_distance'

]



TARGET = 'trip_duration'
def split_dataset(df, features, target):

    X = df[features]

    y = df[target]

    

    return X, y
X_train, y_train = split_dataset(

    df,

    features=FEATURES,

    target=TARGET

)



X_train.shape, y_train.shape
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()
from sklearn.model_selection import cross_val_score



losses = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(-l) for l in losses]



np.mean(losses)
rf.fit(X_train, y_train)



X_test = df_test[FEATURES]
y_test_pred = rf.predict(X_test)

y_test_pred.mean()
df_submission['trip_duration'] = y_test_pred

df_submission.head()
df_submission.to_csv('submission.csv', index=False)
!ls