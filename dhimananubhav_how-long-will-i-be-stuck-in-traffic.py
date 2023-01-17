import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

import random
random.seed(42)
# pretty plots
sns.set_context('talk')
matplotlib.rcParams['font.family'] = 'arial'
df = pd.read_csv('../input/uio_clean.csv')

# convert to datetime and strip seconds 
df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M')
df['dropoff_datetime'] = df['dropoff_datetime'].str.slice(0, 16)
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m-%d %H:%M')

display(df.info())
display(df.head())
original_data_size = df.shape[0]
# creat new feature avg_speed
df['avg_speed'] = df['dist_meters'] / df['trip_duration']  # m/sec
df['avg_speed'] = 3.6*df['avg_speed'] # km/hr
df['avg_speed'] = np.round(df['avg_speed'])

# Remove observations with missing values
df.dropna(how='any', axis='rows', inplace=True)

# Removing observations with erroneous values
mask = df['pickup_longitude'].between(-80, -77)
mask &= df['dropoff_longitude'].between(-80, -77)
mask &= df['pickup_latitude'].between(-4, 1)
mask &= df['dropoff_latitude'].between(-4, 1)
mask &= df['trip_duration'].between(30, 2*3600)
mask &= df['wait_sec'].between(0, 2*3600)
mask &= df['dist_meters'].between(100, 100*1000)
mask &= (df['trip_duration'] > df['wait_sec'])
mask &= df['avg_speed'].between(5, 90)

df = df[mask]
cleaned_data_size = df.shape[0]

print('Original dataset size:', original_data_size,
     '\nRemoving erroneous value. Cleaned dataset size:', cleaned_data_size)
# checking if Ids are unique, 
print("Number of columns and rows and columns are {} and {} respectively.".format(df.shape[1], df.shape[0]))
if df.id.nunique() == df.shape[0]:
    print("Dataset ids are unique")
print('\n')

# check data integrity
# remove observations if trip_duration is off my more than 2 mins of pickup and dropoff timestamp 
df['check_trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).map(lambda x: x.total_seconds())
duration_difference = df[np.abs(df['check_trip_duration'].values  - df['trip_duration'].values) > 2*60]
print('Deleting', duration_difference.shape[0], 'rows with trip_duration different from pickup and dropoff times')
df = df[np.abs(df['check_trip_duration'].values  - df['trip_duration'].values) <= 2*60]
del df['check_trip_duration']
print('Dataset:', df.shape[0])
print('\n')


print('store_and_fwd_flag column has only 1 unique value:', df['store_and_fwd_flag'].unique())
del df['store_and_fwd_flag']
print('Removing feature: store_and_fwd_flag')
# since most trips are attributed to 1 vendor removing vendor id 
# replacing with boolean if vendor in Quito
vendors = pd.DataFrame(df.groupby('vendor_id')['id'].count())
vendors = vendors.reset_index()
vendors['id'] = 100*vendors['id'] / vendors['id'].sum()
vendors = vendors.sort_values('id', ascending=False)
vendors.columns = ['vendor_id', 'trips']

fig = figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 1, 1)
ax1 = sns.barplot(x="trips", y="vendor_id", data=vendors, palette=("Greys_d"))
ax1.set_xlabel('Percentage of trips', weight='bold')
ax1.set_ylabel('Vendor', weight = 'bold')
ax1.set_title('Quito vendor breakdown\n')

sns.despine()
plt.tight_layout();
print(np.round(vendors.trips[0],1),'% of trips have 1 vendor')
df['quito'] = 1*(df['vendor_id'] == 'Quito')

# deleting col 'store_and_fwd_flag'
del df['vendor_id']
print('Removing feature: vendor_id')
print('Adding feature: quito')
# Whats the daily number of trips?
df['pick_date'] = df['pickup_datetime'].dt.date
df['pick_date'] = pd.to_datetime(df['pick_date'])

pickups = pd.DataFrame(df.groupby(['pick_date'])['id'].count())
pickups = pickups.reset_index()
pickups = pickups.sort_values('pick_date', ascending=True)

# inital data is very sparse - remove sparse dates
start_date = pickups[pickups.id >1]['pick_date'].min()
end_date = pickups[pickups.id >1]['pick_date'].max()
#cutoff_date = pd.to_datetime('2017-06-30').date()

print('Dataset starts from', start_date.date(),'till', end_date.date())

# lets start from '2016-06-22' where we have more than 1 daily trip
pickups = pickups[pickups.pick_date >= start_date]

fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 1, 1)
ax1 = sns.lineplot(x="pick_date", y="id", data=pickups)
ax1.set_ylabel('No. of trips', weight = 'bold')
ax1.set_xlabel('Pickup date', weight = 'bold')
ax1.set_title('Daily trips\n')
handles, labels = ax1.get_legend_handles_labels()

sns.despine()
plt.tight_layout();
# quantiles of wait_sec 
# median is around 4 mins
np.quantile(df.wait_sec, [.05, .25, .50, .75, .95])
# source https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial

def haversine_(lat1, lng1, lat2, lng2):
    """haversine distance: 
    great-circle distance between two points on a sphere given their longitudes and latitudes."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)

def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """manhatten distance: 
    sum of  horizontal and vertical distances between points on a grid"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b

import math
def bearing_array(lat1, lng1, lat2, lng2):
    """bearing:
    horizontal angle between direction of an object and another object"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
# distance related features 
# round to nearest meter
df.loc[:,'hvsine_pick_drop'] = haversine_(df['pickup_latitude'].values, 
                                                df['pickup_longitude'].values, 
                                                df['dropoff_latitude'].values, 
                                                df['dropoff_longitude'].values)
df.loc[:,'hvsine_pick_drop'] = np.round(df.loc[:,'hvsine_pick_drop'], 3)
df.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(df['pickup_latitude'].values, 
                                                           df['pickup_longitude'].values, 
                                                           df['dropoff_latitude'].values, 
                                                           df['dropoff_longitude'].values)
df.loc[:,'manhtn_pick_drop'] = np.round(df.loc[:,'manhtn_pick_drop'], 3)
# direction of travel
df.loc[:,'bearing'] = bearing_array(df['pickup_latitude'].values, 
                                          df['pickup_longitude'].values, 
                                          df['dropoff_latitude'].values, 
                                          df['dropoff_longitude'].values)
# different speed based on distance measures
# 1 m/s = 3.6 km/hr 
df.loc[:, 'avg_speed_h'] = 3.6 * 1000 * df['hvsine_pick_drop'] / (df['trip_duration'])
df.loc[:, 'avg_speed_m'] = 3.6 * 1000 * df['manhtn_pick_drop'] / (df['trip_duration'])

# round 
df.loc[:, 'avg_speed_h'] = np.round(df.loc[:, 'avg_speed_h'])
df.loc[:, 'avg_speed_m'] = np.round(df.loc[:, 'avg_speed_m'])
## Equador holiday list
# https://www.officeholidays.com/countries/ecuador/2017.php
holidays = ['2016-01-01', '2016-02-08', '2016-02-09', '2016-03-25', '2016-03-27', 
            '2016-05-01', '2016-05-27', '2016-07-24', '2016-08-10', '2016-10-09', 
            '2016-11-02', '2016-11-03', '2016-12-06', '2016-12-25', 
            '2017-01-01', '2017-02-27', '2017-02-28', '2017-04-14', '2017-04-16', 
            '2017-05-01', '2017-05-24', '2017-07-24', '2017-08-10', '2017-10-09', 
            '2017-11-02', '2017-11-03', '2017-12-06', '2017-12-25']
holidays = pd.to_datetime(holidays)
# was the day a public holiday?
df['holiday'] = 1*(df['pick_date'].isin(holidays))
# time related features
df.loc[:, 'pick_month'] = df['pickup_datetime'].dt.month
df.loc[:, 'hour'] = df['pickup_datetime'].dt.hour
df.loc[:, 'week_of_year'] = df['pickup_datetime'].dt.weekofyear
df.loc[:, 'day_of_year'] = df['pickup_datetime'].dt.dayofyear
df.loc[:, 'day_of_week'] = df['pickup_datetime'].dt.dayofweek

# what more features can be added?
# hourly temp and rainfall would be nice to have - couldnt find a good datasource :( 
# direction api from google - not free but below is the code
# import googlemaps
# from datetime import datetime
# import pandas as pd

# gmaps = googlemaps.Client(key='HELLO WORLD ')

# orig_lat = 40.767937;orig_lng = -73.982155
# dest_lat = 40.765602;dest_lng = -73.964630
# dept_time = pd.to_datetime('2016-09-17 09:32:00')

# gmaps.distance_matrix((orig_lat,orig_lng),(dest_lat,dest_lng), 
#                       departure_time=dept_time, 
#                       mode='driving')["rows"][0]["elements"][0]["duration"]["value"]

# output: travel time in minutes
# transforming continuous features
df['trip_duration_log'] = np.round(np.log1p(df['trip_duration']), 5)
df['dist_meters_log'] = np.round(np.log1p(df['dist_meters']), 5)
df['hvsine_pick_drop_log'] = np.round(np.log1p(df['hvsine_pick_drop']), 5)
df['manhtn_pick_drop_log'] = np.round(np.log1p(df['manhtn_pick_drop']), 5)
display(df.shape[1])
display(df.columns)
# in past runs coordinates had very high feature importance  
# and training error was 3 times lower than test error
# ie- model is memorising
# binning it to 4 decimals (neighbourhood) from 6
df['pickup_longitude'] = np.round(df['pickup_longitude'], 4)
df['pickup_latitude'] = np.round(df['pickup_latitude'], 4)
df['dropoff_longitude'] = np.round(df['dropoff_longitude'], 4)
df['dropoff_latitude'] = np.round(df['dropoff_latitude'], 4)
from sklearn.model_selection import train_test_split
# final feature list
cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'trip_duration', 'dist_meters', 'avg_speed', 'quito', 
        'hvsine_pick_drop', 'manhtn_pick_drop', 'bearing',
        'avg_speed_h', 'avg_speed_m', 'holiday', 'pick_month', 'hour',
        'week_of_year', 'day_of_year', 'day_of_week', 'trip_duration_log',
        'dist_meters_log', 'hvsine_pick_drop_log', 'manhtn_pick_drop_log', 'pick_date']
X = df[cols].copy()
y = np.log1p(df['wait_sec'].copy())

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

print('Dataset size\nTraining:', X_train.shape[0], '\nTesting:', X_test.shape[0])
fig = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(X_train.groupby('pick_date').size(), '-', label='train')
plt.plot(X_test.groupby('pick_date').size(), '-', label='test')
plt.title('Train and test period complete overlap.')
plt.legend(loc=0)
plt.ylabel('No. of trips', weight = 'bold')
plt.xlabel('Pickup date', weight = 'bold')
plt.yscale('log')
sns.despine()
plt.tight_layout();

del X_train['pick_date']
del X_test['pick_date']
fig = figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

sns.distplot(y_train.values, label='train',)
             #kde_kws=dict(cumulative=True))
sns.distplot(y_test.values, label='test',)
             #kde_kws=dict(cumulative=True))
plt.title('Similar distribution of waiting time')
plt.legend(loc=0)
plt.ylabel('density', weight = 'bold')
plt.xlabel('log(seconds+1)', weight = 'bold')
sns.despine()
plt.tight_layout();
fig = figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
ax1 = sns.distplot(X_train.pickup_longitude, rug=True, kde=True) 
ax1.set_xlim([-80, -77])
ax1.set_ylabel('pickup', weight = 'bold')
ax1.set_xlabel('')
ax1.set_yticks([])
ax1.set_xticks([])

plt.subplot(2, 2, 2)
ax2 = sns.distplot(X_train.pickup_latitude, rug=True, kde=True) 
ax2.set_xlim([-4, 1])
ax2.set_xlabel('')
ax2.set_yticks([])
ax2.set_xticks([]) 

plt.subplot(2, 2, 3)
ax3 = sns.distplot(X_train.dropoff_longitude, rug=True, kde=True) 
ax3.set_xlim([-80, -77])
ax3.set_ylabel('dropoff', weight = 'bold')
ax3.set_xlabel('longitude', weight = 'bold')
ax3.set_yticks([])

plt.subplot(2, 2, 4)
ax4 = sns.distplot(X_train.dropoff_latitude, rug=True, kde=True) 
ax4.set_xlim([-4, 1])
ax4.set_xlabel('latitude', weight = 'bold')
ax4.set_yticks([]) 

sns.despine()
plt.tight_layout();
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire

cvs = ds.Canvas(plot_width=2*150, plot_height=4*150,
               x_range=(-78.6,-78.4), y_range=(-0.4,0.0))

agg = cvs.points(X_train, 'pickup_longitude', 'pickup_latitude', ds.count('trip_duration'))
pickup = tf.set_background(tf.shade(agg, cmap=fire),"black", name="Pickup")

agg = cvs.points(X_train, 'dropoff_longitude', 'dropoff_latitude', ds.count('trip_duration'))
dropoff = tf.set_background(tf.shade(agg, cmap=fire),"black", name="Dropoff")

tf.Images(pickup, dropoff)
fig = figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')

plt.subplot(2, 1, 1)
ax1 = sns.boxplot(x=X_train['trip_duration']/60, flierprops=flierprops)
ax1.set_xlabel('', weight = 'bold')
ax1.set_ylabel('trip duration', weight = 'bold')

plt.subplot(2, 1, 2)
ax2 = sns.boxplot(x=np.expm1(y_train.values)/60, flierprops=flierprops)
ax2.set_xlabel('minutes', weight = 'bold')
ax2.set_ylabel('waiting time', weight = 'bold')

sns.despine()
plt.tight_layout();
fig = figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
flierprops = dict(markerfacecolor='0.75', markersize=5, linestyle='none')

plt.subplot(3, 2, 1)
ax1 = sns.boxplot(x=X_train['dist_meters']/1000, flierprops=flierprops)
ax1.set_ylabel('true', weight = 'bold')
ax1.set_xlabel('', weight = 'bold')

plt.subplot(3, 2, 2)
ax2 = sns.boxplot(x='avg_speed', data=X_train, flierprops=flierprops)
ax2.set_ylabel('', weight = 'bold')
ax2.set_xlabel('', weight = 'bold')

plt.subplot(3, 2, 3)
ax3 = sns.boxplot(x=X_train['hvsine_pick_drop'], flierprops=flierprops)
ax3.set_ylabel('haversine', weight = 'bold')
ax3.set_xlabel('', weight = 'bold')

plt.subplot(3, 2, 4)
ax4 = sns.boxplot(x=X_train['avg_speed_h'], flierprops=flierprops)
ax4.set_ylabel('', weight = 'bold')
ax4.set_xlabel('', weight = 'bold')

plt.subplot(3, 2, 5)
ax5 = sns.boxplot(x=X_train['manhtn_pick_drop'], flierprops=flierprops)
ax5.set_ylabel('manhattan', weight = 'bold')
ax5.set_xlabel('distance (km)', weight = 'bold')

plt.subplot(3, 2, 6)
ax6 = sns.boxplot(x='avg_speed_m', data=X_train, flierprops=flierprops)
ax6.set_ylabel('', weight = 'bold')
ax6.set_xlabel('speed (km/hr)', weight = 'bold')

sns.despine()
plt.tight_layout();
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

reg = linear_model.LinearRegression()
cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
cross_val_score(reg, X_train, y_train, cv=cv)
reg.fit(X_train,y_train)
# Predict on testing and training set
lm_y_pred = reg.predict(X_test)

# Report testing RMSE
print(np.sqrt(mean_squared_error(y_test, lm_y_pred)))
import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
def xgb_evaluate(max_depth, gamma,min_child_weight,max_delta_step,subsample,colsample_bytree):
    params = {'eval_metric': 'rmse',
                  'max_depth': int(max_depth),
                  'subsample': subsample,
                  'eta': 0.1,
                  'gamma': gamma,
                  'colsample_bytree': colsample_bytree,   
                  'min_child_weight': min_child_weight ,
                  'max_delta_step':max_delta_step
                 }
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
        
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
%%time
xgb_bo = BayesianOptimization(xgb_evaluate, {
    'max_depth': (2, 12),
    'gamma': (0.001, 10.0),
    'min_child_weight': (0, 20),
    'max_delta_step': (0, 10),
    'subsample': (0.4, 1.0),
    'colsample_bytree' :(0.4, 1.0)})

# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
params = xgb_bo.res['max']['max_params']
print(params)
params['max_depth'] = int(params['max_depth'])
# Train a new model with the best parameters from the search
model2 = xgb.train(params, dtrain, num_boost_round=250)

# Predict on testing and training set
y_pred = model2.predict(dtest)
y_train_pred = model2.predict(dtrain)
# Report testing and training RMSE
print('Test error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Train error:', np.sqrt(mean_squared_error(y_train, y_train_pred)))

# lm 0.6795364412436091
# better than lm 
sns.distplot(y_pred, label='y_hat')
sns.distplot(y_test, label='y')
plt.legend()
sns.despine()
plt.tight_layout();
# actual vs predicted 
sns.jointplot(y_test, y_pred, color="k", space=0, kind = 'kde') #
sns.despine()
plt.tight_layout();
wait_sec_train = np.expm1(y_train).round()
np.quantile(wait_sec_train, [0, 0.05, 0.5, 0.95, 0.99])
wait_sec_true = np.expm1(y_test).round()
wait_sec_predicted = np.expm1(y_pred).round()
# clip very high values
wait_sec_predicted = np.clip(wait_sec_predicted, 0, np.quantile(wait_sec_train, 0.95))

residual = wait_sec_predicted - wait_sec_true

fig = figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 2, 1)
ax1 = sns.distplot(residual)
ax1.set_title('Distribution of residuals\n')
ax1.set_xlabel('')

plt.subplot(1, 2, 2)
ax2 = sns.boxplot(residual, showfliers=False)
ax2.set_xlabel('')

sns.despine()
plt.tight_layout()
print("Mean absolute error:", np.abs(residual).mean())
print("Median absolute error:", np.abs(residual).median())
# feature importance
fig =  plt.figure(figsize = (12,8))
axes = fig.add_subplot(111)
xgb.plot_importance(model2,ax = axes,height =0.5)
sns.despine()
plt.tight_layout()
