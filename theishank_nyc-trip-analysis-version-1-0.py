# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import time

from sklearn.metrics import mean_squared_error as mse



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import training data

train = pd.read_csv('../input/train.csv')

train.head()
train.info()
#Convert store_and_fwd_flag to categorical binary data. Also, datatype is converted to integer

trains = train.copy()

trains.loc[trains.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0

trains.loc[trains.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1

trains['store_and_fwd_flag'] = pd.to_numeric(trains['store_and_fwd_flag'])

trains.info()
#Convert pickup and dropoff date_time to datetime format

trains['pickup_datetime'] = pd.to_datetime(trains.pickup_datetime)

trains['dropoff_datetime'] = pd.to_datetime(trains.dropoff_datetime)
#import datetime as dt

ptime = trains.pickup_datetime.dt.hour*3600+trains.pickup_datetime.dt.minute*60+trains.pickup_datetime.dt.second

%matplotlib inline

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

sns.despine(left=True)

sns.distplot(ptime, axlabel = 'Pickup Time', label = 'Pickup Time', bins = 20, color="r")

trains.loc[:, 'pickup_time'] = ptime

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
trains.head()
trains.loc[(trains.dropoff_datetime - trains.pickup_datetime).dt.total_seconds() != trains.trip_duration]

differ = (trains.dropoff_datetime - trains.pickup_datetime).dt.total_seconds() - trains.trip_duration

sns.distplot(np.log10(differ.loc[((trains.dropoff_datetime - trains.pickup_datetime).dt.total_seconds()- trains.trip_duration)!=0]), axlabel = 'log10(Duration_Difference)')
def haversine_(lat1, lng1, lat2, lng2):

    """function to calculate haversine distance between two co-ordinates"""

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return(h)
lon1 = trains['pickup_longitude']

lat1 = trains['pickup_latitude']

lon2 = trains['dropoff_longitude']

lat2 = trains['dropoff_latitude']

trains['haversine'] = abs(haversine_(lon1,lat1,lon2,lat2))

trains = trains.loc[trains.haversine!=0]
%matplotlib inline

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

sns.despine(left=True)

sns.distplot(np.log10(trains.haversine), axlabel = 'log10(haversine)', label = 'Haversine', bins = 50, color="b")

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
%matplotlib inline

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

sns.despine(left=True)

sns.distplot(np.log10(trains.trip_duration), axlabel = 'log10(Trip Duration)', label = 'Trip Duration', bins = 50, color="b")

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
trains.trip_duration.describe()
trains_drop1 = trains.loc[(1.7<=np.log10(trains.trip_duration)) & (np.log10(trains.trip_duration)<=4.0)]

trains_drop1.trip_duration.describe()
%matplotlib inline

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

sns.despine(left=True)

sns.distplot(np.log10(trains_drop1.trip_duration), axlabel = 'log10(Trip Duration)', label = 'Trip Duration', bins = 50, color="b")

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
trains_drop1.loc[:, 'month'] = trains_drop1['pickup_datetime'].dt.month

trains_drop1.loc[:, 'week_of_year'] = trains_drop1['pickup_datetime'].dt.weekofyear

trains_drop1.loc[:, 'day_of_month'] = trains_drop1['pickup_datetime'].dt.day

trains_drop1.loc[:, 'day_of_week'] = trains_drop1['pickup_datetime'].dt.dayofweek+1

trains_drop1.head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)

sns.despine(left=True)

sns.distplot(trains_drop1['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])

sns.distplot(trains_drop1['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])

sns.distplot(trains_drop1['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])

sns.distplot(trains_drop1['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
objects = trains_drop1['passenger_count'].unique()

y_pos = np.arange(len(objects))

performance = trains_drop1['passenger_count'].value_counts()

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Number of trips')

plt.title('Passenger Count') 

plt.show()
#Dropping the trips where passenger count is zero

trains_drop2 = trains_drop1.loc[trains_drop1.passenger_count != 0]
objects = trains_drop2['passenger_count'].unique()

y_pos = np.arange(len(objects))

performance = trains_drop2['passenger_count'].value_counts()

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Number of trips')

plt.ylabel('Passenger Count')

plt.title('Passenger Count') 

plt.show()
objects = trains_drop2['vendor_id'].unique()

y_pos = np.arange(len(objects))

performance = trains_drop2['vendor_id'].value_counts()

plt.bar(y_pos, performance, align='center', alpha=0.5, color = ['b','r'])

plt.xticks(y_pos, objects)

plt.ylabel('Number of trips')

plt.title('Vendor_ID') 

plt.show()
objects = trains_drop2['store_and_fwd_flag'].unique()

y_pos = np.arange(len(objects))

performance = trains_drop2['store_and_fwd_flag'].value_counts()

plt.bar(y_pos, np.log10(performance), align='center', alpha=0.5, color = ['b','r'])

plt.xticks(y_pos, objects)

plt.ylabel('Number of flags')

plt.title('store_and_fwd_flag') 

plt.show()
trains_drop2.columns
important_features = ['vendor_id', 'pickup_longitude', 'pickup_latitude',

                      'dropoff_longitude', 'dropoff_latitude','pickup_time','haversine',

                      'month','week_of_year','day_of_month','day_of_week']

X = trains_drop2[important_features]

X.info()
y = trains_drop2['trip_duration']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
start=time.time()

from xgboost import XGBRegressor

xgb_pr = XGBRegressor()

xgb_pr.fit(X_train,y_train)

end=time.time()

print(" completed in ",end-start," seconds")
start=time.time()

y_predtrain = xgb_pr.predict(X_test)

end=time.time()

print(" completed in ",end-start," seconds")
y_predtrain
y_tests = np.array(y_test)
len(y_predtrain)
from scipy import stats

differ = abs(y_tests - y_predtrain.astype(int))

stats.describe(differ)
mse(y_tests,y_predtrain.astype(int))
y_predtrain2 = y_predtrain.astype(int).copy()

for index,row in enumerate(y_predtrain2):

    if(row<0):

        y_predtrain2[index]=0

for index,row in enumerate(y_predtrain2):

    if(row<0):

        print(row)
from sklearn.metrics import mean_squared_log_error as msle

msle(y_tests,y_predtrain2)
test = pd.read_csv('../input/test.csv')

tests = test.copy()
tests['pickup_datetime'] = pd.to_datetime(tests.pickup_datetime)

ptime = tests.pickup_datetime.dt.hour*3600+tests.pickup_datetime.dt.minute*60+tests.pickup_datetime.dt.second

tests.loc[:, 'pickup_time'] = ptime

tests.loc[tests.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0

tests.loc[tests.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1

tests['store_and_fwd_flag'] = pd.to_numeric(tests['store_and_fwd_flag'])

lon1 = trains['pickup_longitude']

lat1 = trains['pickup_latitude']

lon2 = trains['dropoff_longitude']

lat2 = trains['dropoff_latitude']

tests['haversine'] = abs(haversine_(lon1,lat1,lon2,lat2))

tests.loc[:, 'month'] = tests['pickup_datetime'].dt.month

tests.loc[:, 'week_of_year'] = tests['pickup_datetime'].dt.weekofyear

tests.loc[:, 'day_of_month'] = tests['pickup_datetime'].dt.day

tests.loc[:, 'day_of_week'] = tests['pickup_datetime'].dt.dayofweek+1

X_tests = tests[important_features]
start=time.time()

y_tpred = xgb_pr.predict(X_tests)

end=time.time()

print(" completed in ",end-start," seconds")
y_tpred
for index,row in enumerate(y_tpred):

    if(row<0):

        print(row)

for index,row in enumerate(y_tpred):

    if(row<0):

        y_tpred[index]=0
df_benchmark = pd.DataFrame()

df_benchmark['id'] = tests['id']

df_benchmark['trip_duration'] = y_tpred
len(y_tpred)
df_benchmark.to_csv("sample_submission5.csv",index=False)