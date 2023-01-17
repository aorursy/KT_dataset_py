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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
# engine='c', parse_dates, infer_datetime_format

train = pd.read_csv('../input/train.csv') 

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
train.info()
train.head()
print(train.columns)

print(test.columns) # dropoff_datetime, trip_duration 칼럼이 없다.
print(np.setdiff1d(train.columns, test.columns))
print('train내 id중복없음 :', train['id'].nunique() == train.shape[0])

# id의 유니크 개수와 행개수가 동일하면 id중복은 없다는 것.

print('train과 test의 id중복없음 :', len(np.intersect1d(train.id.values, test.id.values)) == 0)

# train과 test의 id가 중복없음.
print(train.count().min() == train.shape[0])

print(test.count().min() == test.shape[0])

# missing value없음.
print(train.store_and_fwd_flag.unique())

print(test.store_and_fwd_flag.unique())

print(train['vendor_id'].unique())

print(test['vendor_id'].unique())

# 해당 칼럼은 값이 2개
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

# 문자 라벨 인코딩 Y는 1, N은 0
train['pickup_datetime']= pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime']= pd.to_datetime(train['dropoff_datetime'])

test['pickup_datetime']= pd.to_datetime(test['pickup_datetime'])

# 시간변수를 문자열에서 datetime으로 변환, csv에서 불러올때 바로 변환하는 방법도 있다.
train['pickup_date']= train['pickup_datetime'].dt.date

test['pickup_date']= test['pickup_datetime'].dt.date

# 시,분,초을 제외한 time추출 (2016-05-01)
# 지역정보, 경도/위도에 대해 살펴본다.

train[['pickup_longitude','dropoff_longitude', 'pickup_latitude', 'dropoff_latitude']].describe()
train['idx']= train.index
# longitude 값의 분포는?

fig, ax = plt.subplots(ncols=2, sharey=True)

train.plot.scatter(x='idx', y='pickup_longitude', ax=ax[0])

train.plot.scatter(x='idx', y='dropoff_longitude', ax=ax[1])
# latitude 값의 분포는?

fig, ax = plt.subplots(ncols=2, sharey=True)

train.plot.scatter(x='idx', y='pickup_latitude', ax=ax[0])

train.plot.scatter(x='idx', y='dropoff_latitude', ax=ax[1])
# 위의 그래프를 확인했을때

#  -80 < long < -60

#  32 < lat < 45

# 평균치 73.97 , 40.75 +-0.2 (그래프 확인시)

long_m= -73.97

lat_m= 40.75

city_long_border = (long_m-0.2, long_m+0.2)

city_lat_border = (lat_m-0.2, lat_m+0.2) 

# city_long_border = (-74.03, -73.75)

# city_lat_border = (40.63, 40.85) 
N = 100000

fig,ax= plt.subplots(ncols=2, sharex=True, sharey=True)

ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N], 

              color='b', s=1, label='train', alpha=0.5)

ax[1].scatter(test['pickup_longitude'].values[:N], test['pickup_latitude'].values[:N], 

              color='g', s=1, label='train', alpha=0.5)



plt.xlim(city_long_border) # long

plt.ylim(city_lat_border) # lat

plt.show()

# 다시 조정

city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85) 



N = 100000

fig,ax= plt.subplots(ncols=2, sharex=True, sharey=True)

ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N], 

              color='b', s=1, label='train', alpha=0.5)

ax[1].scatter(test['pickup_longitude'].values[:N], test['pickup_latitude'].values[:N], 

              color='g', s=1, label='train', alpha=0.5)



plt.xlim(city_long_border) # long

plt.ylim(city_lat_border) # lat

plt.show()
print(train['trip_duration'].min())

print(train['trip_duration'].max() / 3600)

train.plot.scatter(x='idx', y='trip_duration')

# 1초??, 979시간?? 아웃밸류임.

# Fortunately the evaluation metric is RMSLE and not RMSE . 

# Outliers will cause less trouble. We could logtransform our target label and use RMSE during training.
# 운행시간 48시간 이상 삭제 

Idx = train[train['trip_duration'] > 48*3600].index

train= train.drop(Idx, axis=0)

# train= train[train['trip_duration'] < 48*3600]
7+8+6+5+23+7+3
# 위도/경도를 PCA값으로 변환 (2차원->2차원)

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

import warnings



warnings.filterwarnings('ignore')
coords= np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

          train[['dropoff_latitude', 'dropoff_longitude']].values,

          test[['pickup_latitude', 'pickup_longitude']].values,

          test[['dropoff_latitude', 'dropoff_longitude']].values         

         ))
coords.shape # 수직으로 쌓다. 행방향으로 쌓았음. 
pca = PCA().fit(coords)

# PCA 주성분 분석

# (1) 특성추출: 특성들이 통계적으로 상관관계가 없도록 데이터셋을 회전하는 기술 (2차원에서 2차원 또는 1차원으로 변환)

# (2) 차원축소 : 고차원의 데이터셋 시각화 (10차원에서 3차원~1차원으로 변환)
# PCA 지역정보

# 기존 4개 칼럼 -> PCA로 변환된 칼럼 4개

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]

train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]

train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
fig, ax = plt.subplots(ncols=2)

ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N],

              color='blue', s=1, alpha=0.1)

ax[1].scatter(train['pickup_pca0'].values[:N], train['pickup_pca1'].values[:N],

              color='green', s=1, alpha=0.1)

ax[0].set_xlim(city_long_border)

ax[0].set_ylim(city_lat_border)

pca_borders = pca.transform([[x, y] for x in city_lat_border for y in city_long_border])

ax[1].set_xlim(pca_borders[:, 0].min(), pca_borders[:, 0].max())

ax[1].set_ylim(pca_borders[:, 1].min(), pca_borders[:, 1].max())

plt.show()
# 클러스터링 지역정보, 2가지 컬럼

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])



train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])

train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])

test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
fig, ax = plt.subplots(ncols=1, nrows=1)

ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,

           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

plt.show()
# 경도/위도 bin, 센터 경도/위도 bin, 지역정보 4가지 (추후 입력특성으로 사용하지 않음.), long경도, lat위도

train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)

test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)

test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
# 거리정보를 구해보자.

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
# 거리 4가지, train

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
# 거리 4가지, test

test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])
# 센터 거리정보 (승차/하차위치의 센터, 승차/하차위치의 센터bin)

# 4가지 컬럼

train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2

test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2



train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)

train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)

test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)

test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
# 시간 (총 6가지)

# 이번주 몇번째날?, 

# 이번년도 몇번째주?,

# 몇시, 몇분, 몇번째초, 이번주 몇번째시, 몇번째초bin?



train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday # 해당 주에 몇번째 일

train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear # 해당 년에 몇번째 주

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds() # 전체 시간을 초로 변환

train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']

train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600)) # dt 초를 3시간 단위로 범주화



test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']

test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600)) 

train[['pickup_datetime','pickup_hour','pickup_minute','pickup_dt', 

       'pickup_weekday', 'pickup_hour_weekofyear','pickup_week_hour', 'pickup_dt_bin']].head()
# 거리(km)/운행시간(s) = 속도(km/s)

# 속도 3가지

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
# 시간대에 따른 속도변화 시각화

fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(15,5))

ax[0].plot(train.groupby('pickup_hour').mean()['avg_speed_h'], 'bo-', lw=2, alpha=0.7)

ax[1].plot(train.groupby('pickup_weekday').mean()['avg_speed_h'], 'go-', lw=2, alpha=0.7)

ax[2].plot(train.groupby('pickup_week_hour').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7)

ax[0].set_xlabel('hour')

ax[1].set_xlabel('weekday')

ax[2].set_xlabel('weekhour')

ax[0].set_ylabel('average speed')

fig.suptitle('Rush hour average traffic speed')

plt.show()
# 지역범위에 따른 속도변화 시각화



train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)

# Average speed for regions

gby_cols = ['pickup_lat_bin', 'pickup_long_bin']

coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()

coord_count = train.groupby(gby_cols).count()[['id']].reset_index()

coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

coord_stats = coord_stats[coord_stats['id'] > 100] # 100개 이상의 값에 대한 평균치만 허용

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,5))

ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N],

           color='black', s=1, alpha=0.5)

ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values,

           c=coord_stats.avg_speed_h.values,

           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)

ax.set_xlim(city_long_border)

ax.set_ylim(city_lat_border)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

plt.title('Average speed')

plt.show()
# Temporal and geospatial aggregation

# 시간과 지역에 따른 평균속도 확인 (18가지)

# 시, 일, 초bin, 주시, 클러스터 승차, 클러스터 하차에 따른 하버사인 평속 (6) 

# 위 동일,에 따른 더미 평속 (6)

# 위 동일,에 따른 운행 시간 (6)



for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',

               'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:

    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]

    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]

    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)

    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

# 시간과 지역에 따른 평균속도 확인 (5가지)

# 센터경도+위도, 시+센터경도+위도, 시+클러스터승차, 시+클러스터하차 (4)

# 클러스터승차+하차에 따른 하버사인 평속 (1)





for gby_cols in [['center_lat_bin', 'center_long_bin'],

                 ['pickup_hour', 'center_lat_bin', 'center_long_bin'],

                 ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],

                 ['pickup_cluster', 'dropoff_cluster']]:

    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()

    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()

    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

    coord_stats = coord_stats[coord_stats['id'] > 100]

    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]

    train = pd.merge(train, coord_stats, how='left', on=gby_cols)

    test = pd.merge(test, coord_stats, how='left', on=gby_cols)
# 운행횟수 (8가지)

# 위 동일, id횟수 (5)

# pickup_datetime_group # 시간을 60분 단위로 그룹화 시킨다.

# dropoff_cluster_count # 해당 클러스터 지역에서 시간당(60분) 하차 횟수

# pikup_cluster_count # 해당 클러스터 지역에서 시간당 승차 횟수



# group_freq = '60min'

# df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]

# train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)

# test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)



# # Count trips over 60min

# df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()

# df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']

# train = train.merge(df_counts, on='id', how='left')

# test = test.merge(df_counts, on='id', how='left')



# # Count how many trips are going to each cluster over time

# dropoff_counts = df_all \

#     .set_index('pickup_datetime') \

#     .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \

#     .agg({'id': 'count'}) \

#     .reset_index().set_index('pickup_datetime') \

#     .groupby('dropoff_cluster').rolling('240min').mean() \

#     .drop('dropoff_cluster', axis=1) \

#     .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \

#     .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})



# train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

# test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)





# Count how many trips are going from each cluster over time

# df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]

# pickup_counts = df_all \

#     .set_index('pickup_datetime') \

#     .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster']) \

#     .agg({'id': 'count'}) \

#     .reset_index().set_index('pickup_datetime') \

#     .groupby('pickup_cluster').rolling('240min').mean() \

#     .drop('pickup_cluster', axis=1) \

#     .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \

#     .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})



# train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)

# test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
# OSMR 피쳐, 외부데이타 (3가지)

# 총 거리, 총 운행시간, 스텝 갯수 (???)

# 승차/하차 위치의 실제도로 최단거리

# 예측운행시간

# fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',

#                   usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])

# fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',

#                   usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

# test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',

#                                usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

# train_street_info = pd.concat((fr1, fr2))

# train = train.merge(train_street_info, how='left', on='id')

# test = test.merge(test_street_info, how='left', on='id')

# train_street_info.head()
# train과 test 컬럼차이 -> 입력특성에서 제외

print(np.setdiff1d(train.columns, test.columns)) 

feature_names = list(train.columns)

print(np.setdiff1d(train.columns, test.columns))

do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',

                           'trip_duration', 'idx',

                           'pickup_date', 'avg_speed_h', 'avg_speed_m',

                           'pickup_lat_bin', 'pickup_long_bin',

                           'center_lat_bin', 'center_long_bin',

                           'pickup_dt_bin', 'pickup_datetime_group']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print('We have %i features.' % len(feature_names))



y = np.log(train['trip_duration'].values + 1)

train.isnull().sum()
import xgboost as xgb



Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model.best_score) # train 0.35, valid 0.385
feature_importance_dict = model.get_fscore()

fs = ['f%i' % i for i in range(len(feature_names))]

f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),

                   'importance': list(feature_importance_dict.values())})

f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})

feature_importance = pd.merge(f1, f2, how='right', on='f')

feature_importance = feature_importance.fillna(0)

feature_importance[['feature_name', 'importance']].sort_values(by='importance',ascending=False)
ypred = model.predict(dvalid)



plt.scatter(ypred, yv, s=0.1, alpha=0.1)

plt.xlabel('log(prediction)')

plt.ylabel('log(ground truth)')

plt.show()
plt.scatter(np.exp(ypred), np.exp(yv), s=0.1, alpha=0.1)

plt.xlabel('prediction')

plt.ylabel('ground truth')

plt.show()
ytest = model.predict(dtest)



test['trip_duration'] = np.exp(ytest) - 1

test[['id', 'trip_duration']].to_csv('submission_v2.csv', index=False)