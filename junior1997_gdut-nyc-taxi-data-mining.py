%matplotlib inline

import pandas as pd

from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import mean_squared_error

from math import radians, cos, sin, asin, sqrt

import seaborn as sns

import matplotlib

import numpy as np

import matplotlib.pyplot as plt
train=pd.read_csv("../input/taxi-trip-duration-train-csv/train.csv")

train.head(3)
test=pd.read_csv("../input/new-york-taxi-trip-duration-test-csv/test.csv")

test.head(3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train.describe()
test.describe()
train.isnull().any()
test.isnull().any()
m = np.mean(train['trip_duration'])

s = np.std(train['trip_duration'])

train = train[train['trip_duration'] <= m + 2*s]

train = train[train['trip_duration'] >= m - 2*s]

west, south, east, north = -74.03, 40.63, -73.77, 40.85



train = train[(train.pickup_latitude> south) & (train.pickup_latitude < north)]

train = train[(train.dropoff_latitude> south) & (train.dropoff_latitude < north)]

train = train[(train.pickup_longitude> west) & (train.pickup_longitude < east)]

train = train[(train.dropoff_longitude> west) & (train.dropoff_longitude < east)]
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)#转成日期格式

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)#转成日期格式

train['pickup_date'] = train['pickup_datetime'].dt.date#新增一列，单独把日期提出来

test['pickup_date'] = test['pickup_datetime'].dt.date#新增一列，单独把日期提取出来

train['Month'] = train['pickup_datetime'].dt.month#新增一列，单独把月提出来

test['Month'] = test['pickup_datetime'].dt.month#新增一列，单独把月提出来

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek#新增一列，单独把周提出来

test['dayofweek'] = test['pickup_datetime'].dt.dayofweek#新增一列，单独把周提出来

train['DayofMonth'] = train['pickup_datetime'].dt.day#新增一列，单独把日提出来

test['DayofMonth'] = test['pickup_datetime'].dt.day#新增一列，单独把日提出来

train['Hour'] = train['pickup_datetime'].dt.hour#新增一列，单独把小时提出来

test['Hour'] = test['pickup_datetime'].dt.hour#新增一列，单独把小时提出来

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)#转成日期格式
plt.hist(train['trip_duration'].values, bins=100)

plt.xlabel('trip_duration')

plt.ylabel('number of train records')

plt.show()
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

plt.hist(train['log_trip_duration'].values, bins=100)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()

sns.distplot(train["log_trip_duration"], bins =100)
#这是每天的接客数量

plt.subplots(1,1,figsize=(16,9))

plt.plot(train.groupby('pickup_date').count()[['id']], 'o-', label='train')

plt.plot(test.groupby('pickup_date').count()[['id']], 'o-', label='test')

plt.title('Trips over Time.')

plt.legend(loc=0)

plt.ylabel('Trips')

plt.show()
fig1 = plt.figure(figsize = (10,6))

ax1 = fig1.add_subplot(2,2,1)

ax1.plot(train.groupby('Month').count()[['id']], 'o-')

ax1.plot(test.groupby('Month').count()[['id']], 'o-')

plt.xlabel('month')

ax2 = fig1.add_subplot(2,2,2)

ax2.plot(train.groupby('dayofweek').count()[['id']], 'o-')

ax2.plot(test.groupby('dayofweek').count()[['id']], 'o-')

plt.xlabel('dayofweek')

ax3 = fig1.add_subplot(2,2,3)

ax3.plot(train.groupby('DayofMonth').count()[['id']], 'o-')

ax3.plot(test.groupby('DayofMonth').count()[['id']], 'o-')

plt.xlabel('dayofmonth')

ax4 = fig1.add_subplot(2,2,4)

ax4.plot(train.groupby('Hour').count()[['id']], 'o-')

ax4.plot(test.groupby('Hour').count()[['id']], 'o-')

plt.xlabel('hour')

plt.show()
#观察出租公司对行程的影响

import warnings

warnings.filterwarnings("ignore")

plot_vendor = train.groupby('vendor_id').mean()['trip_duration']

plt.subplots(1,1,figsize=(8,6))

plt.ylim(ymin=800)

plt.ylim(ymax=840)

sns.barplot(plot_vendor.index,plot_vendor.values)

plt.title('Time per Vendor')

plt.legend(loc=0)

plt.ylabel('Time in Seconds')
month_usage = pd.value_counts(train['Month']).sort_index()

dow_usage = pd.value_counts(train['dayofweek']).sort_index()

hour_usage = pd.value_counts(train['Hour']).sort_index()



#数量

month_vendor = train.groupby(['Month', 'vendor_id']).size().unstack()

dow_vendor = train.groupby(['dayofweek', 'vendor_id']).size().unstack()

hour_vendor = train.groupby(['Hour', 'vendor_id']).size().unstack()



#flag

server = train.groupby(['store_and_fwd_flag', 'vendor_id']).size().unstack()
#横坐标标签

x_tick_labels_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

x_tick_labels_day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']



fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9, 8))



figure = plt.subplot(2, 2, 1)

month_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)

plt.title('Pickups over Month of Year', fontsize = 13)

plt.xlabel('Month', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.xticks(month_usage.index - 1, x_tick_labels_month, rotation='90', fontsize=12)

plt.xticks(rotation=0)



figure = plt.subplot(2, 2, 2)

dow_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)

plt.title('Pickups over Day of Week', fontsize = 13)

plt.xlabel('Day of Week', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.xticks(dow_usage.index, x_tick_labels_day, rotation='90', fontsize=12)

plt.xticks(rotation=0)





figure = plt.subplot(2, 2, 3)

hour_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)

plt.title('Pickups over Hour of Day', fontsize = 13)

plt.xlabel('Hour of Day', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.xticks(rotation=0)



figure = plt.subplot(2, 2, 4)

server.plot.bar(stacked=True, alpha = 0.7, ax = figure)

plt.title('Vehicle Server Access', fontsize = 13)

plt.xlabel(' ', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.xticks(rotation=0)



fig.tight_layout()
#观察flag标记对行程的影响

snwflag = train.groupby('store_and_fwd_flag')['trip_duration'].mean()

plt.subplots(1,1,figsize=(8,6))

plt.ylim(ymin=0)

plt.ylim(ymax=1100)

plt.title('Time per store_and_fwd_flag')

plt.legend(loc=0)

plt.ylabel('Time in Seconds')

sns.barplot(snwflag.index,snwflag.values)
#观察乘客数量对行程的影响

pc = train.groupby('passenger_count')['trip_duration'].mean()

plt.subplots(1,1,figsize=(10,6))

plt.ylim(ymin=0)

plt.ylim(ymax=1100)

plt.title('Time per store_and_fwd_flag')

plt.legend(loc=0)

plt.ylabel('Time in Seconds')

sns.barplot(pc.index,pc.values)
#观察train和test的轨迹 

city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(17.5,8))

ax[0].scatter(train['pickup_longitude'].values[:100000], train['pickup_latitude'].values[:100000],

              color='blue', s=1, label='train', alpha=0.1)

ax[1].scatter(test['pickup_longitude'].values[:100000], test['pickup_latitude'].values[:100000],

              color='green', s=1, label='test', alpha=0.1)

fig.suptitle('Train and test area complete overlap.')

ax[0].set_ylabel('latitude')

ax[0].set_xlabel('longitude')

ax[1].set_xlabel('longitude')

plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.show()
def haversine_array(lat1, lng1, lat2, lng2):#球面距离

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):#曼哈顿距离

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):#轨迹角度

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
train['distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)    

    

train['distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)



train['direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
#观察速度的变化规律

train['avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

train['avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

fig2 = plt.figure(figsize = (16,9))

ax5 = fig2.add_subplot(1,3,1)

ax5.plot(train.groupby('Month').mean()['avg_speed_h'], 'ro-')

plt.xlabel('month')

plt.yticks(np.arange(3, 7, step=0.5))

ax6 = fig2.add_subplot(1,3,2)

ax6.plot(train.groupby('dayofweek').mean()['avg_speed_h'], 'go-')

plt.xlabel('dayofweek')

plt.yticks(np.arange(3, 7, step=0.5))

ax7 = fig2.add_subplot(1,3,3)

ax7.plot(train.groupby('Hour').mean()['avg_speed_h'], 'bo-')

plt.xlabel('hour')

plt.yticks(np.arange(3, 7, step=0.5))

plt.show()
#集群算法，给每个项分配一个集群号。

coord_pickup = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,                  

                          test[['pickup_latitude', 'pickup_longitude']].values))

coord_dropoff = np.vstack((train[['dropoff_latitude', 'dropoff_longitude']].values,                  

                           test[['dropoff_latitude', 'dropoff_longitude']].values))

coords = np.hstack((coord_pickup,coord_dropoff))# 4维

sample_ind = np.random.permutation(len(coords))

kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000).fit(coords[sample_ind])

for df in (train,test):

    df.loc[:, 'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',

                                                         'dropoff_latitude','dropoff_longitude']])
plt.figure(figsize=(16,16))

N = 500

for i in range(10):

    plt.subplot(4,3,i+1)

    tmp_data = train[train.pickup_dropoff_loc==i]

    drop = plt.scatter(tmp_data['dropoff_longitude'][:N], tmp_data['dropoff_latitude'][:N], s=10, lw=0, alpha=0.5,label='dropoff')

    pick = plt.scatter(tmp_data['pickup_longitude'][:N], tmp_data['pickup_latitude'][:N], s=10, lw=0, alpha=0.4,label='pickup')    

    plt.xlim([-74.05,-73.75]);plt.ylim([40.6,40.9])

    plt.legend(handles = [pick,drop])

    plt.title('clusters %d'%i)
#异常值已顺便清除

train.describe()
train = train.drop(['id', 'vendor_id', 'pickup_date','pickup_datetime'], axis = 1)

train = train.drop(['dropoff_datetime','avg_speed_h','avg_speed_m', 'trip_duration'], axis = 1)

Test_id = test['id']

test = test.drop(['id','vendor_id','pickup_date','pickup_datetime'], axis = 1)
train.info()

test.info()
train.describe()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train['store_and_fwd_flag'] = le.fit_transform(train['store_and_fwd_flag'])

test['store_and_fwd_flag'] = le.transform(test['store_and_fwd_flag'])
x = train.drop(['log_trip_duration'],1)

y = train['log_trip_duration']
#随机森林模型，交叉验证计算分数

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

model_rnd_frst=RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, 

                                     min_samples_split=2, min_samples_leaf=1, 

                                     min_weight_fraction_leaf=0.0, max_features='auto', 

                                     max_leaf_nodes=None, min_impurity_split=1e-07, 

                                     bootstrap=True, oob_score=False, n_jobs=-1, 

                                     random_state=None, verbose=1, warm_start=False)

print(cross_val_score(model_rnd_frst,x,y))
#XGBOOST模型

Xtr, Xv, ytr, yv = train_test_split(x, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

#dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



#From beluga's kernel

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 150, watchlist, early_stopping_rounds=100,

                  maximize=False, verbose_eval=10)

print('Modeling RMSLE %.5f' % model.best_score)
#调参方法

#md = [6]

#lr = [0.1,0.3]

#mcw = [20,25,30]

#for m in md:

#    for l in lr:

#        for n in mcw:

#            t0 = datetime.now()

#            xgb_pars = {'min_child_weight': mcw, 'eta': lr, 'colsample_bytree': 0.9, 

#                        'max_depth': md,

#            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

#            'eval_metric': 'rmse', 'objective': 'reg:linear'}

#            model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,

#                  max
xgb.plot_importance(model)
dtest = xgb.DMatrix(test)

pred = model.predict(dtest)

pred = np.exp(pred) - 1
'''submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)

submission.columns = ['id','trip_duration']

submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)

submission.to_csv("submission.csv", index=False)'''
#submission.head(5)