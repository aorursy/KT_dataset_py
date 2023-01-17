# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import time

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_log_error as msle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import training data

train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

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
trains.info()
#import datetime as dt

ptime = trains.pickup_datetime.dt.hour*100+trains.pickup_datetime.dt.minute+trains.pickup_datetime.dt.second*0.01

# %matplotlib inline

# sns.set(style="white", palette="muted", color_codes=True)

# f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

# sns.despine(left=True)

# sns.distplot(ptime, axlabel = 'Pickup Time', label = 'Pickup Time', bins = 20, color="r")

# plt.setp(axes, yticks=[])

# plt.tight_layout()

# plt.show()
trains.loc[:, 'pickup_time'] = ptime
# differ = (trains.dropoff_datetime - trains.pickup_datetime).dt.total_seconds() - trains.trip_duration

# sns.distplot(np.log10(differ.loc[((trains.dropoff_datetime - trains.pickup_datetime).dt.total_seconds()- trains.trip_duration)!=0]))
def haversine_(lat1, lng1, lat2, lng2):

    """function to calculate haversine distance between two co-ordinates"""

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return(h)
# %matplotlib inline

# sns.set(style="white", palette="muted", color_codes=True)

# f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

# sns.despine(left=True)

# sns.distplot(np.log10(trains.trip_duration), axlabel = 'log10(Trip Duration)', label = 'Trip Duration', bins = 50, color="b")

# plt.setp(axes, yticks=[])

# plt.tight_layout()

# plt.show()
# trains.trip_duration.describe()
# %matplotlib inline

# sns.set(style="white", palette="muted", color_codes=True)

# f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

# sns.despine(left=True)

# sns.distplot(np.log10(trains_drop1.trip_duration), axlabel = 'log10(Trip Duration)', label = 'Trip Duration', bins = 50, color="b")

# plt.setp(axes, yticks=[])

# plt.tight_layout()

# plt.show()
# sns.set(style="white", palette="muted", color_codes=True)

# f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)

# sns.despine(left=True)

# sns.distplot(trains_drop1['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])

# sns.distplot(trains_drop1['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])

# sns.distplot(trains_drop1['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])

# sns.distplot(trains_drop1['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])

# plt.setp(axes, yticks=[])

# plt.tight_layout()

# plt.show()
# objects = trains_drop1['vendor_id'].unique()

# y_pos = np.arange(len(objects))

# performance = trains['vendor_id'].value_counts()

# plt.bar(y_pos, performance, align='center', alpha=0.5, color = ['b','r'])

# plt.xticks(y_pos, objects)

# plt.ylabel('Number of trips')

# plt.title('Vendor_ID') 

# plt.show()
# objects = trains_drop1['store_and_fwd_flag'].unique()

# y_pos = np.arange(len(objects))

# performance = trains['store_and_fwd_flag'].value_counts()

# plt.bar(y_pos, np.log10(performance), align='center', alpha=0.5, color = ['b','r'])

# plt.xticks(y_pos, objects)

# plt.ylabel('Number of flags')

# plt.title('store_and_fwd_flag') 

# plt.show()
trains.info()
trains.loc[trains.trip_duration<=120]
new_trains=trains.loc[(trains.trip_duration>=120) & (trains.trip_duration<=32400)]
new_trains.head()
lon1 = new_trains['pickup_longitude']

lat1 = new_trains['pickup_latitude']

lon2 = new_trains['dropoff_longitude']

lat2 = new_trains['dropoff_latitude']

new_trains['haversine'] = haversine_(lon1,lat1,lon2,lat2)
new_trains=new_trains.loc[new_trains.haversine!=0]
new_trains.loc[:, 'pick_month'] = new_trains['pickup_datetime'].dt.month

new_trains.loc[:, 'week_of_year'] = new_trains['pickup_datetime'].dt.weekofyear

new_trains.loc[:, 'day_of_year'] = new_trains['pickup_datetime'].dt.dayofyear

new_trains.loc[:, 'day_of_week'] = new_trains['pickup_datetime'].dt.dayofweek
# new_trains.columns
X_train_ann = new_trains[['vendor_id','pickup_longitude', 'pickup_latitude',

                       'dropoff_longitude', 'dropoff_latitude','pickup_time','haversine', 'pick_month',

       'week_of_year', 'day_of_year', 'day_of_week']]

X_train_ann.describe()
y_train_ann = new_trains['trip_duration']

y_train_ann.describe()
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train_ann, y_train_ann, test_size=0.2, random_state=60)


nn = MLPRegressor(

hidden_layer_sizes=(10,2),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',

learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



              
start=time.time()

n = nn.fit(X_train_1, y_train_1)

end=time.time()

print("Time Taken In Fitting: ",end-start)
rough = nn.predict(X_test_1)

msle(y_test_1,rough)
# train_osrm_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')

# train_osrm_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

# train_osrm = pd.concat([train_osrm_1, train_osrm_2])

# train_osrm = train_osrm[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]

# train_df = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')

# train_osrm = pd.merge(train_df, train_osrm, on = 'id', how = 'left')
# train_osrm.info()
# train_osrm.loc[train_osrm.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0

# train_osrm.loc[train_osrm.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1

# train_osrm['store_and_fwd_flag'] = pd.to_numeric(train_osrm['store_and_fwd_flag'])

# train_osrm['pickup_datetime'] = pd.to_datetime(train_osrm.pickup_datetime)

# train_osrm['dropoff_datetime'] = pd.to_datetime(train_osrm.dropoff_datetime)



# ptime = train_osrm.pickup_datetime.dt.hour*100+train_osrm.pickup_datetime.dt.minute+train_osrm.pickup_datetime.dt.second*0.01

# train_osrm.loc[:, 'pickup_time'] = ptime
# new_train_osrm=train_osrm.loc[(train_osrm.trip_duration>=120) & (train_osrm.trip_duration<=32400)]
# lon1 = new_train_osrm['pickup_longitude']

# lat1 = new_train_osrm['pickup_latitude']

# lon2 = new_train_osrm['dropoff_longitude']

# lat2 = new_train_osrm['dropoff_latitude']

# new_train_osrm['haversine'] = haversine_(lon1,lat1,lon2,lat2)

# new_train_osrm=new_train_osrm.loc[new_train_osrm.haversine!=0]

# new_train_osrm.loc[:, 'pick_month'] = new_train_osrm['pickup_datetime'].dt.month

# new_train_osrm.loc[:, 'week_of_year'] = new_train_osrm['pickup_datetime'].dt.weekofyear

# new_train_osrm.loc[:, 'day_of_year'] = new_train_osrm['pickup_datetime'].dt.dayofyear

# new_train_osrm.loc[:, 'day_of_week'] = new_train_osrm['pickup_datetime'].dt.dayofweek
# new_train_osrm=new_train_osrm.loc[(new_train_osrm.passenger_count!=0)]
# new_train_osrm.columns
# X_train_ann = new_train_osrm[['vendor_id','pickup_longitude', 'pickup_latitude',

#                        'dropoff_longitude', 'dropoff_latitude','total_distance',

#        'pickup_time','haversine', 'pick_month',

#        'week_of_year', 'day_of_year', 'day_of_week']]

# y_train_ann = new_train_osrm['trip_duration']

# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train_ann, y_train_ann, test_size=0.2, random_state=60)


# nn = MLPRegressor(

# hidden_layer_sizes=(10,2),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',

# learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,

# random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

# early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



              
# start=time.time()

# n = nn.fit(X_train_1, y_train_1)

# end=time.time()

# print("Time Taken In Fitting: ",end-start)
# rough = nn.predict(X_test_1)

# msle(y_test_1,rough)

# rough
test=pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

test.head()
test.loc[test.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0

test.loc[test.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1

test['store_and_fwd_flag'] = pd.to_numeric(test['store_and_fwd_flag'])

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

ptime_test = test.pickup_datetime.dt.hour*100+test.pickup_datetime.dt.minute+test.pickup_datetime.dt.second*0.01

test.loc[:, 'pickup_time'] = ptime_test
test.info()
lon1 = test['pickup_longitude']

lat1 = test['pickup_latitude']

lon2 = test['dropoff_longitude']

lat2 = test['dropoff_latitude']

test['haversine'] = haversine_(lon1,lat1,lon2,lat2)

test.loc[:, 'pick_month'] = test['pickup_datetime'].dt.month

test.loc[:, 'hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'week_of_year'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'day_of_year'] = test['pickup_datetime'].dt.dayofyear

test.loc[:, 'day_of_week'] = test['pickup_datetime'].dt.dayofweek
X_test_prediction=test[['vendor_id','pickup_longitude', 'pickup_latitude',

                       'dropoff_longitude', 'dropoff_latitude','pickup_time','haversine', 'pick_month',

       'week_of_year', 'day_of_year', 'day_of_week']]
X_test_prediction.info()
y_test_predicted=nn.predict(X_test_prediction)
df_benchmark = pd.DataFrame()

df_benchmark['id'] = test['id']

df_benchmark['trip_duration'] = y_test_predicted
y_test_predicted
df_benchmark.to_csv("sample_submission2.csv",index=False)
# test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

# test_fr = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')

# test_fr_new = test_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]

# test_df = pd.merge(test_df, test_fr_new, on = 'id', how = 'left')

# test_df.head()
# test_df.loc[test_df.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0

# test_df.loc[test_df.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1

# test_df['store_and_fwd_flag'] = pd.to_numeric(test_df['store_and_fwd_flag'])



# test_df['pickup_datetime'] = pd.to_datetime(test_df.pickup_datetime)

# ptime = test_df.pickup_datetime.dt.hour*100+test_df.pickup_datetime.dt.minute+test_df.pickup_datetime.dt.second*0.01

# test_df.loc[:, 'pickup_time'] = ptime
# lon1 = test_df['pickup_longitude']

# lat1 = test_df['pickup_latitude']

# lon2 = test_df['dropoff_longitude']

# lat2 = test_df['dropoff_latitude']

# test_df['haversine'] = haversine_(lon1,lat1,lon2,lat2)

# test_df=test_df.loc[test_df.haversine!=0]

# test_df.loc[:, 'pick_month'] = test_df['pickup_datetime'].dt.month

# test_df.loc[:, 'week_of_year'] = test_df['pickup_datetime'].dt.weekofyear

# test_df.loc[:, 'day_of_year'] = test_df['pickup_datetime'].dt.dayofyear

# test_df.loc[:, 'day_of_week'] = test_df['pickup_datetime'].dt.dayofweek
# X_test_prediction=test_df[['vendor_id','pickup_longitude', 'pickup_latitude',

#                        'dropoff_longitude', 'dropoff_latitude','total_distance',

#        'pickup_time','haversine', 'pick_month',

#        'week_of_year', 'day_of_year', 'day_of_week']]
# y_test_predicted=nn.predict(X_test_prediction)
# y_test_predicted
# df_benchmark = pd.DataFrame()

# df_benchmark['id'] = test_df['id']

# df_benchmark['trip_duration'] = y_test_predicted

# df_benchmark.to_csv("sample_submission3.csv",index=False)