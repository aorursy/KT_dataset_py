# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('dark_background')

sns.set_style("darkgrid")

%%time

# Reading File

train_path  = '../input/train.csv'



# Set columns to most suitable type to optimize for memory usage, default is float 64 but we just need float 32, it will save a lot of RAM

traintypes = {'fare_amount': 'float32',

              'pickup_datetime': 'str', 

              'pickup_longitude': 'float32',

              'pickup_latitude': 'float32',

              'dropoff_longitude': 'float32',

              'dropoff_latitude': 'float32',

              'passenger_count': 'uint8'}



cols = list(traintypes.keys())

# I used 2.000.000 rows to test and 10.000.000 to commit

train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows=2_000_000)
%%time

# Save into feather format, it will be faster for the next time 

train_df.to_feather('nyc_taxi_data_raw.feather')
# load the same dataframe next time directly, without reading the csv file again!



df_train = pd.read_feather('nyc_taxi_data_raw.feather')

# check datatypes

df_train.dtypes
# check statistics of the features

df_train.describe()
len(df_train[df_train.fare_amount > 0])
# Since they are less than 300 records so we will drop them ( for 2.000.000 rows)

df_train = df_train[df_train.fare_amount>=0]
# IQR is 3.5 so we can see most data less than 20 but we will plot data to 100 to see some outliers

sns.distplot(df_train[df_train.fare_amount < 100].fare_amount, bins=50);
# Count the number of null value

df_train.isnull().sum()
# it's not too much so we will drop all row with null value

df_train = df_train.dropna(how = 'any', axis = 'rows')

# read test data

df_test =  pd.read_csv('../input/test.csv')

df_test.head(5)
df_test.describe()
# We will paste pickup date to date type

df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
df_train['pickup_datetime']
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
def add_new_date_time_features(dataset):

    dataset['hour'] = dataset.pickup_datetime.dt.hour

    dataset['day'] = dataset.pickup_datetime.dt.day

    dataset['month'] = dataset.pickup_datetime.dt.month

    dataset['year'] = dataset.pickup_datetime.dt.year

    dataset['day_of_week'] = dataset.pickup_datetime.dt.dayofweek

    

    return dataset
df_train = add_new_date_time_features(df_train)

df_test = add_new_date_time_features(df_test)
df_train.describe()


df_train = df_train[df_train.pickup_longitude.between(df_test.pickup_longitude.min(), df_test.pickup_longitude.max())]

df_train = df_train[df_train.pickup_latitude.between(df_test.pickup_latitude.min(), df_test.pickup_latitude.max())]

df_train = df_train[df_train.dropoff_longitude.between(df_test.dropoff_longitude.min(), df_test.dropoff_longitude.max())]

df_train = df_train[df_train.dropoff_latitude.between(df_test.dropoff_latitude.min(), df_test.dropoff_latitude.max())]
df_train.shape
def calculate_abs_different(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

calculate_abs_different(df_train)

calculate_abs_different(df_test)
# Since we are calculating this at New York, we can assign a constant, rather than using a formula

# longitude = degrees of latitude in radians * 69.172

#1 degree of longitude = 50 miles

def convert_different_miles(df):

    df['abs_diff_longitude'] = df.abs_diff_longitude*50

    df['abs_diff_latitude'] = df.abs_diff_latitude*69

convert_different_miles(df_train)

convert_different_miles(df_test)
### Angle difference between north, and manhattan roadways

meas_ang = 0.506 # 29 degrees = 0.506 radians (https://en.wikipedia.org/wiki/Commissioners%27_Plan_of_1811)

import math





## adding extra features

def add_distance(df):

    df['Euclidean'] = (df.abs_diff_latitude**2 + df.abs_diff_longitude**2)**0.5 ### as the crow flies  

    df['delta_manh_long'] = (df.Euclidean*np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()

    df['delta_manh_lat'] = (df.Euclidean*np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()

    df['distance'] = df.delta_manh_long + df.delta_manh_lat

    df.drop(['abs_diff_longitude', 'abs_diff_latitude','Euclidean', 'delta_manh_long', 'delta_manh_lat'], axis=1, inplace=True)

add_distance(df_train)

add_distance(df_test)
df_train.head()
def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    """

    Return distance along great radius between pickup and dropoff coordinates.

    """

    #Define earth radius (km)

    R_earth = 6371

    #Convert degrees to radians

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = pickup_lon - dropoff_lon

    

    #Compute bearing distance

    a = np.arctan2(np.sin(dlon * np.cos(dropoff_lat)),np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon))

    return a
# We will convert pandas to numpy to get the best performance

train_df['direction'] = calculate_direction(train_df['pickup_latitude'].values, train_df['pickup_longitude'].values, 

                                   train_df['dropoff_latitude'].values , train_df['dropoff_longitude'].values) 

df_test['direction'] = calculate_direction(df_test['pickup_latitude'].values, df_test['pickup_longitude'].values, 

                                   df_test['dropoff_latitude'].values , df_test['dropoff_longitude'].values) 
train_df['pickup_latitude'].apply(lambda x: np.radians(x))

train_df['pickup_longitude'].apply(lambda x: np.radians(x))

train_df['dropoff_latitude'].apply(lambda x: np.radians(x))

train_df['dropoff_longitude'].apply(lambda x: np.radians(x))



df_test['pickup_latitude'].apply(lambda x: np.radians(x))

df_test['pickup_longitude'].apply(lambda x: np.radians(x))

df_test['dropoff_latitude'].apply(lambda x: np.radians(x))

df_test['dropoff_longitude'].apply(lambda x: np.radians(x))
sns.jointplot(x='distance', y='fare_amount', data=df_train)

# We can see that distance is less than 100 so it makes sense and we can use it
# We extracted feature with day, week, month, year so we can remove pickup_datetime

df_train.drop(columns=['pickup_datetime'], inplace=True)



y = df_train['fare_amount']

df_train = df_train.drop(columns=['fare_amount'])
df_train.head()
from sklearn.model_selection import train_test_split

import lightgbm as lgbm

x_train,x_test,y_train,y_test = train_test_split(df_train,y,random_state=123,test_size=0.1)
params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': 4,

        'num_leaves': 31,

        'learning_rate': 0.1,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 10000 ,

        'bagging_freq': 10,

        'metric': 'rmse',  

        'zero_as_missing': True,

        'num_rounds':50000

    }
train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','day','day_of_week'])

valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','day','day_of_week'])

model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=1000,verbose_eval=500, valid_sets=valid_set)
df_train.describe()
test_key = df_test['key']



df_test.drop(columns=["pickup_datetime",'key'], axis=1, inplace=True)
prediction = model.predict(df_test, num_iteration = model.best_iteration)      

submission = pd.DataFrame({

        "key": test_key,

        "fare_amount": prediction

})



submission.to_csv('taxi_fare_submission.csv',index=False)