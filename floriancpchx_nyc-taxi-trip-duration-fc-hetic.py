# Importing Python libraries for data analysis, processing, modelling and visualization.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading data and checking it

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
# check desired output

df_sample = pd.read_csv('../input/sample_submission.csv')

df_sample.head()
# Checking whether ID is unique or not.

len(df_train.index) == df_train.index.nunique()
len(df_test.index) == df_test.index.nunique()
# Checking for null values

df_train.isnull().values.any()
df_test.isnull().values.any()
# Quick analysis.

df_train.describe()
df_test.describe()
df_train.info()
df_test.info()
# Changing data type to handle dates in a easier way

df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
#Plotting trip duration

df_train['log_trip_duration'] = np.log(df_train['trip_duration'].values + 1)

plt.hist(df_train['log_trip_duration'].values,bins=50)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()
# Splitting the dates in several columns to find out if any particulars months/days/hours/minutes have a different behavior 

df_train.loc[:, 'pickup_month'] = df_train['pickup_datetime'].dt.month

df_train.loc[:, 'pickup_weekday'] = df_train['pickup_datetime'].dt.weekday

df_train.loc[:, 'pickup_day'] = df_train['pickup_datetime'].dt.day

df_train.loc[:, 'pickup_hour'] = df_train['pickup_datetime'].dt.hour

df_train.loc[:, 'pickup_minute'] = df_train['pickup_datetime'].dt.minute



df_train.loc[:, 'dropoff_month'] = df_train['dropoff_datetime'].dt.month

df_train.loc[:, 'dropoff_weekday'] = df_train['dropoff_datetime'].dt.weekday

df_train.loc[:, 'dropoff_day'] = df_train['dropoff_datetime'].dt.day

df_train.loc[:, 'dropoff_hour'] = df_train['dropoff_datetime'].dt.hour

df_train.loc[:, 'dropoff_minute'] = df_train['dropoff_datetime'].dt.minute





df_test.loc[:, 'pickup_month'] = df_test['pickup_datetime'].dt.month

df_test.loc[:, 'pickup_weekday'] = df_test['pickup_datetime'].dt.weekday

df_test.loc[:, 'pickup_day'] = df_test['pickup_datetime'].dt.day

df_test.loc[:, 'pickup_hour'] = df_test['pickup_datetime'].dt.hour

df_test.loc[:, 'pickup_minute'] = df_test['pickup_datetime'].dt.minute

# Creating a fonction to calculate distance between our lat/long pickup and dropoff coordinates.

from math import sin, cos, sqrt, atan2, radians



def calculate_distance(longitude1, latitude1, longitude2, latitude2):

  #  Radius of Earth in km

    R = 6373.0



    lat1 = radians(latitude1)

    lon1 = radians(longitude1)

    lat2 = radians(latitude2)

    lon2 = radians(longitude2)



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Return distance in km

    return R * c



def add_distance(df):

    return df.apply(lambda row: calculate_distance(

         row.pickup_longitude,

         row.pickup_latitude,

         row.dropoff_longitude,

         row.dropoff_latitude

     ), axis=1)



df_train['distance_km'] = add_distance(df_train)

df_test['distance_km'] = add_distance(df_test)

df_train.head()
# Categorical Data treatment



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df_train['store_and_fwd_flag'])

df_train['store_and_fwd_flag'] = le.transform(df_train['store_and_fwd_flag'])

df_test['store_and_fwd_flag'] = le.transform(df_test['store_and_fwd_flag'])



le.fit(df_train['vendor_id'])

df_train['vendor_id'] = le.transform(df_train['vendor_id'])

df_test['vendor_id'] = le.transform(df_test['vendor_id'])
# Variables we will be using to train

train_variables = ["vendor_id","passenger_count","store_and_fwd_flag","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","pickup_month","pickup_weekday","pickup_day","pickup_hour",'pickup_minute',"distance_km"]
y = np.log1p(df_train['trip_duration'])

df_train["trip_duration"] 

X = df_train[train_variables]

X.shape, y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=1337)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=1337,n_jobs=-1,bootstrap=True,n_estimators=20)

fit = rf.fit(X_train, y_train)
#Pray for good fortune

score = rf.score(X_valid, y_valid)

print(score)
test_columns = X_train.columns

predictions = rf.predict(df_test[test_columns])
submission = pd.DataFrame({'id': df_test.id, 'trip_duration': np.expm1(predictions)})

submission.head()
submission.to_csv("submission_florian_coupechoux_nyc_taxi.csv", index=False)
