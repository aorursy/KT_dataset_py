import gc
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-white')
cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
train = pd.read_csv('../input/train.csv', usecols=cols, engine='c')
test = pd.read_csv('../input/test.csv', usecols=cols[1:], engine='c')
float_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
train[float_cols] = np.round(train[float_cols].astype('float16'), 2)
test[float_cols] = np.round(test[float_cols].astype('float16'), 2)
gc.collect()
train['year'] = train.pickup_datetime.str[:4].astype('uint16')
test['year'] = test.pickup_datetime.str[:4].astype('uint16')
gc.collect()

train['month'] = train.pickup_datetime.str[5:7].astype('uint8')
test['month'] = test.pickup_datetime.str[5:7].astype('uint8')
gc.collect()

train['day'] = train.pickup_datetime.str[8:10].astype('uint8')
test['day'] = test.pickup_datetime.str[8:10].astype('uint8')
gc.collect()

train['hour'] = train.pickup_datetime.str[11:13].astype('uint8')
test['hour'] = test.pickup_datetime.str[11:13].astype('uint8')
gc.collect()

# train['minute'] = train.pickup_datetime.str[14:16].astype('uint8')
# test['minute'] = test.pickup_datetime.str[14:16].astype('uint8')
# gc.collect()

# train['second'] = train.pickup_datetime.str[17:19].astype('uint8')
# test['second'] = test.pickup_datetime.str[17:19].astype('uint8')
# gc.collect()

train = train.drop('pickup_datetime', axis=1)
test = test.drop('pickup_datetime', axis=1)
gc.collect()
len(train)
len(test)
train.head()
test.head()
# train = train.sample(frac=0.25).reset_index(drop=True)
# gc.collect()
Y = train.fare_amount
Y.value_counts().head(10)
from sklearn.metrics import median_absolute_error, make_scorer
print('MAE Error (in USD):', median_absolute_error(Y, np.ones_like(Y.values) * 6.5))
print('MAE Error (in USD):', median_absolute_error(Y, np.ones_like(Y.values) * 4.5))
_ = Y.plot.hist(100, color='teal')
_ = np.log1p(Y).plot.hist(bins=100, color='teal')
# Credit: https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
# commented out due to slowness
# from scipy.stats import lognorm
# shape, loc, scale = lognorm.fit(Y.sample(1000000).values, loc=0)

shape, loc, scale = 0.6009456315880513, -0.01272662981718287, 9.165648701197032
print('--Log-normal Distribution--')
print('\tmu:', np.log(scale))
print('\tsigma:', shape)
print('\t[linear space / in dollars] mu:', scale)
print('\t[linear space / in dollars] sigma:', np.exp(shape))
_ = np.round(Y[(Y>=0)&(Y<=30)]).astype(int).hist(bins=30, color='teal')
np.round(Y.describe(), 2)
mean = Y.mean()
print('Arithmetic mean:', mean)
print('MAE Error (in USD):', median_absolute_error(Y, mean * np.ones_like(Y)))
median = Y.median()
print('Median:', median)
print('MAE Error (in USD):', median_absolute_error(Y, median * np.ones_like(Y)))
geom_mean = np.expm1(np.log1p(Y).mean())
print('Geometric mean:', geom_mean)
print('MAE Error (in USD):', median_absolute_error(Y, geom_mean * np.ones_like(Y)))
pickup_map = train.groupby(['pickup_longitude', 'pickup_latitude'])['fare_amount'].mean().reset_index()
pickup_map = pickup_map.pivot('pickup_longitude', 'pickup_latitude', 'fare_amount').fillna(0)
pickup_map = pickup_map.loc[pickup_map.index[~np.isinf(pickup_map.index)], 
               pickup_map.columns[~np.isinf(pickup_map.columns)]]
gc.collect()
_ = plt.contour(
    pickup_map.columns, 
    pickup_map.index, 
    np.log1p(pickup_map.values), 
    cmap='viridis')
_ = plt.colorbar()
pickup_map = train.groupby(['pickup_longitude', 'pickup_latitude'])['fare_amount'].mean().reset_index()
pickup_map = pickup_map.pivot('pickup_longitude', 'pickup_latitude', 'fare_amount').fillna(0)
pickup_map = pickup_map.loc[pickup_map.index[(~np.isinf(pickup_map.index)) & (pickup_map.index>-76) & (pickup_map.index<-72)], 
               pickup_map.columns[~np.isinf(pickup_map.columns) & (pickup_map.columns>39) & (pickup_map.columns<42.25)]]
gc.collect()
_ = plt.contour(
    pickup_map.columns, 
    pickup_map.index, 
    np.log1p(pickup_map.values), 
    cmap='viridis')
_ = plt.colorbar()
train['pickup_central'] = ((train.pickup_longitude>-76) & (train.pickup_longitude<-72) & (train.pickup_latitude>39) & (train.pickup_latitude<42.25)).astype(int)
train.pickup_central.value_counts(True)
test['pickup_central'] = ((test.pickup_longitude>-76) & (test.pickup_longitude<-72) & (test.pickup_latitude>39) & (test.pickup_latitude<42.25)).astype(int)
test.pickup_central.value_counts(True)
dropoff_map = train.groupby(['dropoff_longitude', 'dropoff_latitude'])['fare_amount'].mean().reset_index()
dropoff_map = dropoff_map.pivot('dropoff_longitude', 'dropoff_latitude', 'fare_amount').fillna(0)
dropoff_map = dropoff_map.loc[dropoff_map.index[~np.isinf(dropoff_map.index)], 
               dropoff_map.columns[~np.isinf(dropoff_map.columns)]]
gc.collect()
_ = plt.contour(
    dropoff_map.columns, 
    dropoff_map.index, 
    np.log1p(dropoff_map.values), 
    cmap='viridis')
_ = plt.colorbar()
dropoff_map = train.groupby(['dropoff_longitude', 'dropoff_latitude'])['fare_amount'].mean().reset_index()
dropoff_map = dropoff_map.pivot('dropoff_longitude', 'dropoff_latitude', 'fare_amount').fillna(0)
dropoff_map = dropoff_map.loc[dropoff_map.index[(~np.isinf(dropoff_map.index)) & (dropoff_map.index>-76) & (dropoff_map.index<-72)], 
               dropoff_map.columns[~np.isinf(dropoff_map.columns) & (dropoff_map.columns>39) & (dropoff_map.columns<42.25)]]
gc.collect()
_ = plt.contour(
    dropoff_map.columns, 
    dropoff_map.index, 
    np.log1p(dropoff_map.values), 
    cmap='viridis')
_ = plt.colorbar()
train['dropoff_central'] = ((train.dropoff_longitude>-76) & (train.dropoff_longitude<-72) & (train.dropoff_latitude>39) & (train.dropoff_latitude<42.25)).astype(int)
train.dropoff_central.value_counts(True)
test['dropoff_central'] = ((test.dropoff_longitude>-76) & (test.dropoff_longitude<-72) & (test.dropoff_latitude>39) & (test.dropoff_latitude<42.25)).astype(int)
test.dropoff_central.value_counts(True)
n_train = len(train)
train = train[(train['pickup_central']==1)].reset_index(drop=True)
train = train[(train['dropoff_central']==1)].reset_index(drop=True)
train = train.drop(['pickup_central','dropoff_central'],axis=1)
gc.collect()
print('Remaining:', len(train)/n_train)
from sklearn.tree import DecisionTreeRegressor
X = train[['pickup_longitude', 'pickup_latitude']]
Y = train.fare_amount.values
idx = np.isfinite(X.pickup_longitude) & np.isfinite(X.pickup_latitude) & np.isfinite(Y)
X = X[idx]
Y = Y[idx]
print(cross_val_score(DecisionTreeRegressor(min_samples_split=30000), X, Y))
m = DecisionTreeRegressor(min_samples_split=30000).fit(X, Y)
X['prediction'] = np.log1p(m.predict(X))
_ = X.groupby(['pickup_longitude', 'pickup_latitude']).prediction.mean().reset_index().plot.scatter('pickup_longitude', 'pickup_latitude', c='prediction', s=1, cmap='viridis')
del X, Y, m, idx
gc.collect()
X = train[['dropoff_longitude', 'dropoff_latitude']]
Y = train.fare_amount.values
idx = np.isfinite(X.dropoff_longitude) & np.isfinite(X.dropoff_latitude) & np.isfinite(Y)
X = X[idx]
Y = Y[idx]
print(cross_val_score(DecisionTreeRegressor(min_samples_split=30000), X, Y))
m = DecisionTreeRegressor(min_samples_split=30000).fit(X, Y)
X['prediction'] = np.log1p(m.predict(X))
_ = X.groupby(['dropoff_longitude', 'dropoff_latitude']).prediction.mean().reset_index().plot.scatter('dropoff_longitude', 'dropoff_latitude', c='prediction', s=1, cmap='viridis')
del X, Y, m, idx
gc.collect()
pickups = train[['pickup_longitude', 'pickup_latitude']].values
dropoffs = train[['dropoff_longitude', 'dropoff_latitude']].values
train['distance'] = np.sqrt(np.square(pickups - dropoffs).sum(axis=1)).round(5)
del pickups, dropoffs
gc.collect()
_ = train.groupby('distance').fare_amount.median().reset_index().plot.scatter('distance', 'fare_amount', s=5)
pickups = train[['pickup_longitude', 'pickup_latitude']].values
dropoffs = train[['dropoff_longitude', 'dropoff_latitude']].values
train['distance'] = np.abs(pickups - dropoffs).sum(axis=1).round(5)
del pickups, dropoffs
gc.collect()
_ = train.groupby('distance').fare_amount.median().reset_index().plot.scatter('distance', 'fare_amount', s=5)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
X, Y = train.distance.values, train.fare_amount.values
idx = np.isfinite(X) & np.isfinite(Y)
X = X[idx].reshape(-1, 1)
Y = Y[idx]
gc.collect()
cross_val_score(Ridge(), X, Y, scoring=make_scorer(median_absolute_error))
from sklearn.isotonic import IsotonicRegression
cross_val_score(IsotonicRegression(3, 140), X.flatten(), Y, scoring=make_scorer(median_absolute_error))
print('[Train] Percentage of rides with 1-6 passengers:', (train.passenger_count.isin(range(1,6))).sum()/len(train))
print('[Test] Percentage of rides with 1-6 passengers:', (test.passenger_count.isin(range(1,6))).sum()/len(test))
train.passenger_count = train.passenger_count.clip(1, 6)
test.passenger_count = test.passenger_count.clip(1, 6)
_ = train.groupby('passenger_count').fare_amount.mean().reset_index().plot.scatter('passenger_count', 'fare_amount')
X, Y = train[['passenger_count']].values, train.fare_amount.values
gc.collect()
cross_val_score(Ridge(), X, Y, scoring=make_scorer(median_absolute_error))
cross_val_score(IsotonicRegression(3, 140), X.flatten(), Y.flatten(), scoring=make_scorer(median_absolute_error))
monthly_fare = train.groupby(['year', 'month']).fare_amount.mean().reset_index()
monthly_fare.index = pd.to_datetime(monthly_fare.year.astype(str) + '-' + monthly_fare.month.astype(str).str.zfill(2) + '-01')
gc.collect()
_ = monthly_fare.fare_amount.plot()
_ = [plt.axvline(pd.to_datetime(m), linestyle='dashed', color='skyblue') for m in monthly_fare.index if m.month==1]
_ = plt.axvspan(pd.to_datetime('2012-08-01'), pd.to_datetime('2012-09-01'), color='pink')
train['hike_status'] = 'pre-hike'
train.loc[(train.year >= 2012) & (train.month >= 9), 'hike_status'] = 'post-hike'

test['hike_status'] = 'pre-hike'
test.loc[(test.year >= 2012) & (test.month >= 9), 'hike_status'] = 'post-hike'
train.groupby('hike_status').fare_amount.median()
years = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
for y in years:
    year_data = train[train.year==y].reset_index(drop=True).groupby('month').fare_amount.mean().reset_index()
    year_data.fare_amount.plot.line(marker='.')
    gc.collect()
_ = plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
_ = plt.axvspan(-0.5, 2.5, color='pink')
_ = plt.legend(years)
train['fare_season'] = 'high fare season'
train.loc[train.month <= 3, 'fare_season'] = 'low fare season'

test['fare_season'] = 'high fare season'
test.loc[test.month <= 3, 'fare_season'] = 'low fare season'
train.groupby('fare_season').fare_amount.median()
hourly_fare = train.groupby('hour').fare_amount.mean().reset_index()
# UTC time adjustment to NYC (GMT-4)
hourly_fare.hour -= 4
hourly_fare.hour %= 24
hourly_fare = hourly_fare.sort_values('hour').reset_index(drop=True)
gc.collect()
_ = hourly_fare.fare_amount.plot.line(marker='.')
hours = ['12mn'] + [str(i) + 'am' for i in range(1,12)] + ['12nn'] + [str(i) + 'pm' for i in range(1,12)]
_ = plt.xticks(range(24), hours)
_ = plt.axhline(11.5, color='pink', linestyle='dashed')
_ = plt.axhline(11, color='pink', linestyle='dashed')
hours_high = set([0, 1, 2, 10, 11, 12, 13, 18])
hours_low = set([3, 4, 5, 6, 17, 19, 20])

train['hourly_seasonality'] = 'normal'
train.loc[(train.hour-4).isin(hours_high), 'hourly_seasonality'] = 'peak'
train.loc[(train.hour-4).isin(hours_low), 'hourly_seasonality'] = 'off-peak'
gc.collect()

test['hourly_seasonality'] = 'normal'
test.loc[(test.hour-4).isin(hours_high), 'hourly_seasonality'] = 'peak'
test.loc[(test.hour-4).isin(hours_low), 'hourly_seasonality'] = 'off-peak'
gc.collect()
train.groupby('hourly_seasonality').fare_amount.mean().sort_values(ascending=False)
# factors = [
#     'passenger_count',
#     'fare_season',
#     'hike_status',
#     'hourly_seasonality',
# ]
# train['segment'] = train[factors].astype(str).T.apply(lambda x: ':'.join(x))
# gc.collect()
# train.segment.value_counts()
# segments = sorted(train.segment.unique())
# scores = {}
# for segment in segments:
#     X = train[train.segment == segment]['distance'].values.flatten()
#     Y = train[train.segment == segment]['fare_amount'].values.flatten()
#     idx = np.isfinite(X) & np.isfinite(Y)
#     X = X[idx]
#     Y = Y[idx]
#     cv_score = cross_val_score(IsotonicRegression(out_of_bounds='clip'), X, Y, scoring=make_scorer(median_absolute_error))
#     scores[segment] = cv_score.mean()
#     print(segment, cv_score)
#     gc.collect()