# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from datetime import datetime
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
pd.set_option('display.float_format', lambda x: '%.3f' %x)

train.describe()
train.info()
duration_mean=train['trip_duration'].mean()

duration_std=train['trip_duration'].std()

train=train[train['trip_duration']<=duration_mean+2*duration_std]

train=train[train['trip_duration']>=duration_mean-2*duration_std]
train = train[train['pickup_longitude'] <= -73.75]

train = train[train['pickup_longitude'] >= -74.03]

train = train[train['pickup_latitude'] <= 40.85]

train = train[train['pickup_latitude'] >= 40.63]

train = train[train['dropoff_longitude'] <= -73.75]

train = train[train['dropoff_longitude'] >= -74.03]

train = train[train['dropoff_latitude'] <= 40.85]

train = train[train['dropoff_latitude'] >= 40.63]
train['pickup_datetime']=pd.to_datetime(train.pickup_datetime)

test['pickup_datetime']=pd.to_datetime(test.pickup_datetime)

train['dropoff_datetime']=pd.to_datetime(train.dropoff_datetime)

train.loc[:,'pickup_date']=train['pickup_datetime'].dt.date

test.loc[:,'pickup_date']=test['pickup_datetime'].dt.date
train.head()
plt.hist(train['trip_duration'],bins=100)

plt.xlabel('trip duration')

plt.ylabel('number of train records')

plt.show()
train['log_trip_duration']=np.log(train['trip_duration'])

plt.hist(train['log_trip_duration'],bins=100)

plt.xlabel('log trip duration')

plt.ylabel('number of train records')

plt.show()
sns.distplot(train['log_trip_duration'],bins=100)

plt.plot(train['id'].groupby(train['pickup_date']).count(),'o-',label='train')

plt.plot(test['id'].groupby(test['pickup_date']).count(),'o-',label='test')

plt.xlabel('pickup date')

plt.ylabel('trip number')

plt.show()
plot_vendor=train.groupby(train['vendor_id'])['trip_duration'].mean()

sns.barplot(plot_vendor.index,plot_vendor.values)

plt.ylabel('time in seconds')

plt.ylim([800,900])
plot_store=train.groupby(train['store_and_fwd_flag'])['trip_duration'].mean()

sns.barplot(plot_store.index,plot_store.values)

plt.ylabel('time in seconds')

plt.ylim([600,1200])
plot_passenger=train.groupby(train['passenger_count'])['trip_duration'].mean()

sns.barplot(plot_passenger.index,plot_passenger.values)

plt.ylabel('time in seconds')

plt.ylim([0,900])
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

fig,ax=plt.subplots(ncols=2,sharex=True,sharey=True)

ax[0].scatter(train['pickup_longitude'].values[:100000],train['pickup_latitude'].values[:100000],color='blue',alpha=0.1,s=1,label='train')

ax[1].scatter(test['pickup_longitude'].values[:100000],test['pickup_latitude'].values[:100000],color='green',alpha=0.1,s=1,label='test')

ax[0].set_ylabel('pickup_latitude')

ax[1].set_xlabel('pickup_longitude')

ax[0].set_xlabel('pickup_longitude')

plt.ylim(city_lat_border)

plt.xlim(city_long_border)
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)   

train.loc[:,'distance_dummy_manhattan']=dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)



train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
train.head()
coord=np.vstack((train[['pickup_latitude','pickup_longitude']].values,train[['dropoff_latitude','dropoff_longitude']].values))
sample_ind=np.random.permutation(len(coord))[:500000]

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import mean_squared_error

kmeans=MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coord[sample_ind])
train.loc[:,'pickup_cluster']=kmeans.predict(train[['pickup_latitude','pickup_longitude']])

train.loc[:,'dropoff_cluster']=kmeans.predict(train[['dropoff_latitude','dropoff_longitude']])

test.loc[:,'pickup_cluster']=kmeans.predict(test[['pickup_latitude','pickup_longitude']])

test.loc[:,'dropoff_cluster']=kmeans.predict(test[['dropoff_latitude','dropoff_longitude']])

fig,ax=plt.subplots(ncols=1,nrows=1)

ax.scatter(train.pickup_longitude[:500000],train.pickup_latitude[:500000],c=train.pickup_cluster[:500000].values,cmap='autumn',alpha=0.2,s=10,lw=0)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('longitude')

ax.set_ylabel('latitude')

plt.show()
train['month']=train['pickup_datetime'].dt.month

test['month']=test['pickup_datetime'].dt.month
train['day']=train['pickup_datetime'].dt.day

test['day']=test['pickup_datetime'].dt.day

train['hour']=train['pickup_datetime'].dt.hour

test['hour']=test['pickup_datetime'].dt.hour

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek

test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
train
vendor_train=pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')

vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')

passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')

passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')

store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')

cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')

cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')

cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')



month_train = pd.get_dummies(train['month'], prefix='m', prefix_sep='_')

month_test = pd.get_dummies(test['month'], prefix='m', prefix_sep='_')

dom_train = pd.get_dummies(train['day'], prefix='dom', prefix_sep='_')

dom_test = pd.get_dummies(test['day'], prefix='dom', prefix_sep='_')

hour_train = pd.get_dummies(train['hour'], prefix='h', prefix_sep='_')

hour_test = pd.get_dummies(test['hour'], prefix='h', prefix_sep='_')

dow_train = pd.get_dummies(train['dayofweek'], prefix='dow', prefix_sep='_')

dow_test = pd.get_dummies(test['dayofweek'], prefix='dow', prefix_sep='_')
passenger_count_test=passenger_count_test.drop('pc_9',axis=1)
train.columns
train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','month','day','hour','dayofweek',

                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)

Test_id = test['id']

test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','month','day','hour','dayofweek',

                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)
train = train.drop(['dropoff_datetime', 'trip_duration'],axis=1)

train = train.drop(['pickup_datetime','pickup_date','avg_speed_h','avg_speed_m'], axis = 1)

test = test.drop(['pickup_datetime','pickup_date'], axis = 1)
Train_Master = pd.concat([train,

                          vendor_train,

                          passenger_count_train,

                          store_and_fwd_flag_train,

                          cluster_pickup_train,

                          cluster_dropoff_train,

                         month_train,

                         dom_train,

                          hour_train,

                          dow_train

                         ], axis=1)
Test_master = pd.concat([test, 

                         vendor_test,

                         passenger_count_test,

                         store_and_fwd_flag_test,

                         cluster_pickup_test,

                         cluster_dropoff_test,

                         month_test,

                         dom_test,

                          hour_test,

                          dow_test], axis=1)
test.columns
training, testing = train_test_split(Train_Master[0:100000], test_size = 0.2)
X_train = training.drop(['log_trip_duration'], axis=1)

Y_train = training["log_trip_duration"]

X_test = testing.drop(['log_trip_duration'], axis=1)

Y_test = testing["log_trip_duration"]



Y_test = Y_test.reset_index().drop('index',axis = 1)

Y_train = Y_train.reset_index().drop('index',axis = 1)
dtrain=xgb.DMatrix(X_train,label=Y_train)

dvalid = xgb.DMatrix(X_test, label=Y_test)

dtest = xgb.DMatrix(Test_master)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
md=[6]

lr=[0.1,0.3]

mcw = [20,25,30]

for m in md:

    for l in lr:

        for n in mcw:

            t0 = datetime.now()

            xgb_pars = {'min_child_weight': n, 'eta': l, 'colsample_bytree': 0.9, 

                        'max_depth': m,

            'subsample': 0.9, 'lambda': 1., 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}

            model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,

                  maximize=False, verbose_eval=1)
model.best_score
pred=model.predict(dtest)

pred = np.exp(pred)

submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)

submission.columns = ['id','trip_duration']

submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)

submission.shape
submission.to_csv("submission.csv", index=False)