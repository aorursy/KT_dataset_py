# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dask.dataframe as dd
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')

import os
TRAIN_PATH = '../input/train.csv'
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
# Counting the number of rows in dataframe

# Same method but more 'pythonic'
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

n_rows = file_len(TRAIN_PATH)
print (f'Exact number of rows: {n_rows}')
# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
%%time

# using panda read_csv to read the entire file in one shot
# df = pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes)
train = pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, nrows = 5000000)
# df.head()
train.head()
train.describe()
# Can see above that there is somehow an entry with 208? passengers. A plane? Haha. Also, somehow a negative fare. 
# Let's get rid of entries with negative fares and passengers that are too large.
print("old size: %d" % len(train))
train = train[train.fare_amount >=0]
print("New size: %d" % len(train))
# check missing data
train.isnull().sum()
# Delete the null lat/long entries.
print("old size: %d" % len(train))
train = train.dropna(how='any', axis=0)
print("New size after dropping missing value: %d" % len(train))
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

plt.style.use('seaborn-whitegrid')
# checking for passenger count greater than 7
train[train.passenger_count >7].passenger_count.hist(bins=10, figsize = (16,8))
plt.xlabel("Passenger Count")
plt.ylabel("Frequency")
# Plotting boxplots for <= 7 passengers.
plt.figure(figsize= (16,8))
sns.boxplot(x = train[train.passenger_count< 8].passenger_count, y = train.fare_amount)
# Looking at correlation of fare_amount and passenger_count - wondering why there are 0 passengers for entries
train[train.passenger_count <7][['fare_amount','passenger_count']].corr()
# Time to look over test file
test = pd.read_csv("../input/test.csv")
print("shape of test data", test.shape)
test.head()
test.isnull().sum()
# No null data - how nice of our test set
test.describe()
# Test set min/max dropoffs and pickups
min(test.pickup_longitude.min(),test.dropoff_longitude.min()), \
max(test.pickup_longitude.max(),test.dropoff_longitude.max())
min(test.pickup_latitude.min(),test.dropoff_latitude.min()), \
max(test.pickup_latitude.max(),test.dropoff_latitude.max())
# this function will also be used with the test set below
def select_within_test_boundary(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
BB = (-74.5, -72.8, 40.5, 41.8)
print('Old size: %d' % len(train))
train = train[select_within_test_boundary(train, BB)]
print('New size: %d' % len(train))
# Going through time features
def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
#     df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
#     df['day_of_year'] = df.pickup_datetime.dt.dayofyear
#     df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
#     df["quarter"] = df.pickup_datetime.dt.quarter
#     df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df
train = prepare_time_features(train)
test = prepare_time_features(test)
# calculate distance between two latitude, longitude points haversine (spherical) formula 
# Returns distance in miles
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))   # 2*R*asin...
train['distance_miles'] = distance(train.pickup_latitude, train.pickup_longitude, \
                                      train.dropoff_latitude, train.dropoff_longitude)
test['distance_miles'] = distance(test.pickup_latitude, test.pickup_longitude, \
                                      test.dropoff_latitude, test.dropoff_longitude)
# Airport distances for NYC
def transform(data):
    # Distances to nearby airports, 
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)

    data['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    return data

train = transform(train)
test = transform(test)
# train[(train['distance_miles']==0)&(train['fare_amount']==0)]
# Deleting the distances & fares equal to 0.
print("old size: %d" % len(train))
train = train.drop(index= train[(train['distance_miles']==0)&(train['fare_amount']==0)].index, axis=0)
print("New size: %d" % len(train))
# How many with $0 fares but distances > 0
train[train['fare_amount']==0].shape
print("old size: %d" % len(train))
train = train.drop(index= train[train['fare_amount']==0].index, axis=0)
print("New size: %d" % len(train))
# Initial costs, so these aren't typical
train[train['fare_amount'] < 2.5].shape
print("old size: %d" % len(train))
train = train.drop(index= train[train['fare_amount'] < 2.5].index, axis=0)
print("New size: %d" % len(train))
# Deleting all remaining cases greater than 7 passengers
print("old size: %d" % len(train))
train = train.drop(index= train[train.passenger_count >= 7].index, axis=0)
print("New size: %d" % len(train))
# What's distribution of distances that are covered now?
pd.cut(train['distance_miles'],np.linspace(0, 70, num = 8)).value_counts()
# Now the test cases
pd.cut(test['distance_miles'],np.linspace(0, 70, num = 8)).value_counts()
# Starting to get to the data set to train/test split
train.columns
test.columns
# create copy of the data set
df_train = train.drop(columns= ['pickup_datetime'], axis= 1).copy()
df_test = test.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
print(df_train.shape)
print(df_test.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('fare_amount', axis=1),
                                                    df_train['fare_amount'], test_size=0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import xgboost as xgb
params = {
    'max_depth': 7,
    'gamma' :0,
    'eta':.03, 
    'subsample': 1,
    'colsample_bytree': 0.9, 
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 0
}
def XGBmodel(X_train,X_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(X_train,label=y_train)
    matrix_test = xgb.DMatrix(X_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(X_train,X_test,y_train,y_test,params)
prediction = model.predict(xgb.DMatrix(df_test), ntree_limit = model.best_ntree_limit).tolist()
test = pd.read_csv("../input/test.csv")
holdout = pd.DataFrame({'key': test['key'], 'fare_amount': prediction})
holdout.to_csv('xgb_with4mtrainedrows.csv', index=False)
# del df
# testdata = pd.read_csv('../input/test.csv')
# testdata.head()
# traindata = pd.read_csv('../input/train.csv')
# traindata.head()
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.impute import SimpleImputer
# my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
#my_pipeline.fit(train_X, train_y)
# predictions = my_pipeline.predict(test_X)
#predictions