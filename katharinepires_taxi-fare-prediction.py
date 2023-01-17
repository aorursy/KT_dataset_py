import pandas as pd
import numpy as np
import random
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime
train = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv')
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
train.shape
train.head(10)
train.dtypes
train = train.sample(n = 250000)
train.shape
train.to_csv('train_sample.csv', index = False)
train = pd.read_csv('./train_sample.csv', parse_dates = ['key', 'pickup_datetime'])
train.shape
train.head()
train.dtypes
train.isna().sum(axis = 0)
key = train['key']
#visualizing data distributions with 100 divisions(bins)::
plt.hist(key, bins = 100)
pdt = train['pickup_datetime']
key.describe()
pdt.describe()
#eliminating seconds:
pdt = pdt.map(lambda date: date.tz_localize(None))
pdt
#key with same format as pdt:
key = pd.to_datetime(key.dt.strftime('%Y-%m-%d %H:%M:%S'))
key
(key == pdt)
(key == pdt).value_counts()
train.drop(['key'], axis = 1, inplace = True)
train.head()
#formatting:
train['pickup_datetime'] = pdt
train.head()
plon = train['pickup_longitude']
plon.describe()
plt.hist(plon, bins = 150)
plon.median()
#A zoom:
plt.hist(plon[(plon > plon.median() - 2.5) & (plon < plon.median() + 2.5)], bins = 100)
##Let's eliminate the inconsistent values:
plon_val = (plon > plon.median() - 0.5) & (plon < plon.median() + 0.5)
plon_val.value_counts()
#percent:
print(plon_val.value_counts(), '\n', plon_val.value_counts(normalize = True))
plat = train['pickup_latitude']
plat.describe()
plt.hist(plat, bins = 100)
plat.median()
plt.hist(plat[(plat > plat.median() - 2.5) & (plat < plat.median() + 2.5)], bins = 100)
plat_val = (plat > plat.median() - 0.5) & (plat < plat.median() + 0.5)
#percent:
print(plat_val.value_counts(), '\n', plat_val.value_counts(normalize = True))
dlon = train['dropoff_longitude']
dlon.describe()
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
ax[0].hist(dlon, bins = 100)
ax[1].hist(dlon[(dlon > dlon.median() - 2.5) & (dlon < dlon.median() + 2.5)], bins = 100)
dlon_val = (dlon > dlon.median() - 0.5) & (dlon < dlon.median() + 0.5)
#percent
print(dlon_val.value_counts(), '\n', dlon_val.value_counts(normalize = True))
dlat = train['dropoff_latitude']
dlat.describe()
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
ax[0].hist(dlat, bins = 100)
ax[1].hist(dlat[(dlat > dlat.median() - 2.5) & (dlat < dlat.median() + 2.5)], bins = 100)
dlat_val = (dlat > dlat.median() - 0.5) & (dlat < dlat.median() + 0.5)
#percent
print(dlat_val.value_counts(), '\n', dlat_val.value_counts(normalize = True))
pcount = train['passenger_count']
pcount.describe()
print((pcount == 0).sum(), (pcount == 0).mean())
plt.hist(pcount, bins = 100)
plt.hist(pcount[pcount < 15], bins = 100)
pcount_val = (pcount >= 1) & (pcount <= 6)
print(pcount_val.value_counts(), '\n', pcount_val.value_counts(normalize = True))
fare = train['fare_amount']
fare.describe()
plt.hist(fare, bins = 100)
plt.hist(fare[fare < 10], bins = 100)
plt.hist(fare[fare > 50], bins = 100)
fare_val = (fare > 2) & (fare < 150)
print(fare_val.value_counts(), '\n', fare_val.value_counts(normalize = True))
#concatenating the values:
val_entries = fare_val & plon_val & plat_val & dlon_val & dlat_val & pcount_val
print(val_entries.value_counts(), '\n', val_entries.value_counts(normalize = True))
train = train.drop(val_entries[val_entries == False].index)
train.head()
train.shape
train['hour_of_day'] = train['pickup_datetime'].map(lambda date: date.timetuple().tm_hour)
train['day_of_week'] = train['pickup_datetime'].map(lambda date: date.timetuple().tm_wday)
train['day_of_year'] = train['pickup_datetime'].map(lambda date: date.timetuple().tm_yday)
train['year'] = train['pickup_datetime'].map(lambda date: date.timetuple().tm_year)
train.head()
len(train['pickup_datetime'][train['pickup_datetime'].dt.strftime('%m-%d') == '02-29'])
train.drop(train['pickup_datetime'][train['pickup_datetime'].dt.strftime('%m-%d') == '02-29'].index, inplace = True)
train.shape
condition = (train['year'] == 2012) & (train['day_of_year'] > 59)
train['day_of_year'][condition] = train['day_of_year'] - 1
fig, ax = plt.subplots(1, 4, figsize = (15,3))
train_not_2015 = train[train['year'] < 2015] # we won't use 2015 because we only have the dates until half of the year
ax[0].hist(train_not_2015['hour_of_day'], bins = 24) # 24 hours in a day
ax[0].set_title('Hour of day')
ax[1].hist(train_not_2015['day_of_week'], bins = 7) # 7 days in a week
ax[1].set_title('Days of week')
ax[2].hist(train_not_2015['day_of_year'], bins = 365) # 365 days
ax[2].set_title('Day of year')
ax[3].hist(train_not_2015['year'], bins = 6) # we have 6 years
ax[3].set_title('Year')
plt.figure(figsize = (15, 5))
plt.scatter(train['pickup_datetime'], train['fare_amount'], s = 1, alpha = 0.2)
from collections import Counter
fare_zoom = train['fare_amount'][(train['fare_amount'] > 40) & (train['fare_amount'] < 60)]
common_fares_zoom = Counter(fare_zoom)
common_fares_zoom
most_common_fares_zoom = common_fares_zoom.most_common(10)
most_common_fares_zoom
plt.bar([x[0] for x in most_common_fares_zoom], [x[1] for x in most_common_fares_zoom])
from mpl_toolkits.basemap import Basemap
#NYC latitude and longitude definition
lat1, lat2 = 40.55, 40.95
lon1, lon2 = -74.10, -73.70

plt.figure(figsize = (10, 10))
m = Basemap(projection = 'cyl', resolution = 'h',
            llcrnrlat = lat1, urcrnrlat = lat2,
            llcrnrlon = lon1, urcrnrlon = lon2)
m.drawcoastlines()
m.fillcontinents(color = 'palegoldenrod', lake_color = 'lightskyblue')
m.drawmapboundary(fill_color = 'lightskyblue')
m.drawparallels(np.arange(lat1, lat2 + 0.05, 0.1), labels = [1, 0, 0, 0])
m.drawmeridians(np.arange(lon1, lon2 + 0.05, 0.1), labels = [0, 0, 0, 1])

#Pickup locations - of all exits (green)
m.scatter(train['pickup_longitude'], train['pickup_latitude'], s = 1, c = 'green',
          alpha = 0.1, zorder = 5)
#Dropoof locations - of all exits (yellow)
m.scatter(train['dropoff_longitude'], train['dropoff_latitude'], s = 1, c='yellow',
         alpha = 0.1, zorder = 5)
for i in [0, 1, 2, 4]:
  this_fare = most_common_fares_zoom[i][0]
  this_df = train[train['fare_amount'] == this_fare]
  #pickup location - red
  m.scatter(this_df['pickup_longitude'], this_df['pickup_latitude'], s = 2, c = 'red',
           alpha = 0.2, zorder = 5)
  #dropoff location - blue
  m.scatter(this_df['dropoff_longitude'], this_df['dropoff_latitude'], s = 2, c = 'blue',
           alpha = 0.2, zorder = 5)
#Arrival point coordinates
coords = train[['dropoff_latitude',
                'dropoff_longitude']][(train['fare_amount'] > 40) &
                                       (train['fare_amount'] < 60) &
                                       (train['dropoff_latitude'] < 40.7) &
                                       (train['dropoff_latitude'] > 40.6) &
                                       (train['dropoff_longitude'] < -73.7) &
                                       (train['dropoff_longitude'] > -73.9)]
coords.shape
coords.head()
print(coords['dropoff_latitude'].median(), coords['dropoff_longitude'].median())
#Starting point coordinates
coords = train[['dropoff_latitude',
                'dropoff_longitude']][(train['fare_amount'] > 40) &
                                       (train['fare_amount'] < 60) &
                                       (train['dropoff_latitude'] < 40.85) &
                                       (train['dropoff_latitude'] > 40.7) &
                                       (train['dropoff_longitude'] < -73.9) &
                                       (train['dropoff_longitude'] > -74.1)]
print(coords['dropoff_latitude'].median(), coords['dropoff_longitude'].median())
filtered = train[['fare_amount', 
                  'passenger_count']][((train['fare_amount'] == most_common_fares_zoom[0][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[1][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[2][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[4][0]))&
                                       (train['fare_amount'] < 60) &
                                       (train['dropoff_latitude'] < 40.7) & 
                                       (train['dropoff_latitude'] > 40.6) & 
                                       (train['dropoff_longitude'] < -73.7) &
                                       (train['dropoff_longitude'] > -73.9)]
plt.scatter(filtered['passenger_count'], filtered['fare_amount'])
filtered2 = train[['fare_amount', 
                  'hour_of_day',
                  'day_of_week',
                  'day_of_year']][((train['fare_amount'] == most_common_fares_zoom[0][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[1][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[2][0]) |
                                       (train['fare_amount'] == most_common_fares_zoom[4][0]))&
                                       (train['fare_amount'] < 60) &
                                       (train['dropoff_latitude'] < 40.7) & 
                                       (train['dropoff_latitude'] > 40.6) & 
                                       (train['dropoff_longitude'] < -73.7) &
                                       (train['dropoff_longitude'] > -73.9)]

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
ax[0].scatter(filtered2['hour_of_day'], filtered2['fare_amount'])
ax[0].set_title('Hour of day')
ax[1].scatter(filtered2['day_of_week'], filtered2['fare_amount'])
ax[1].set_title('Day of week')
ax[2].scatter(filtered2['day_of_year'], filtered2['fare_amount'])
ax[2].set_title('Day of year')
train.drop('pickup_datetime', axis = 1, inplace = True)
train.head()
#conversion to radians:
lon1, lon2 = np.radians(train['pickup_longitude']), np.radians(train['dropoff_longitude'])
lat1, lat2 = np.radians(train['pickup_latitude']), np.radians(train['dropoff_latitude'])
#subtraction from the start point to the end point:
dlon = lon2 - lon1
dlat = lat2 - lat1
#Euclidean Distance (Km)
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
train['eucl_distance'] = 6373 * c

train.head()
#Manhattan Distance (Km)
a1 = np.sin(dlon / 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
a2 = np.sin(dlat / 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train['manh_distance'] = 6373 * (c1 + c2)

train.head()
fig, ax = plt.subplots(1, 2, figsize = (10,5))
ax[0].hist(train['eucl_distance'])
ax[0].set_title('Euclidian Distance')
ax[1].hist(train['manh_distance'])
ax[1].set_title('Manhattan Distance')
X_train = train.drop('fare_amount', axis = 1)
Y_train = train['fare_amount']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
from sklearn.model_selection import cross_val_score
cv = 3
scoring = 'neg_mean_squared_error'
import multiprocessing
n_jobs = multiprocessing.cpu_count() - 1
#Linear Regression:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Ridge Regression:
from sklearn.linear_model import Ridge
model = Ridge()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Lasso Regression:
from sklearn.linear_model import Lasso
model = Lasso()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Nearest Neighbors:
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Decision Tree:
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Random Forest:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Deep Learning:
from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
scores = cross_val_score(model, X_train_scaled, Y_train, cv = cv,
                         scoring = scoring, n_jobs = n_jobs)
np.sqrt(-scores.mean())
from sklearn.model_selection import train_test_split
X_train1, X_test, Y_train1, Y_test = train_test_split(X_train_scaled, Y_train,test_size = 0.2, random_state = 24)
model = RandomForestRegressor(n_estimators = 150)
model.fit(X_train1, Y_train1)

features_importances = model.feature_importances_
argsort = np.argsort(features_importances) #making a ordering of importances
features_importances_sorted = features_importances[argsort]
feature_names = X_train.columns
features_sorted = feature_names[argsort]
plt.barh(features_sorted, features_importances_sorted)
Y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mse
rmse
print_every = int(250000 / 1000)

fig = plt.figure(figsize=(20,5))
plt.bar(list(range(len(Y_test[::print_every]))), Y_test.values[::print_every],
        alpha = 1, color = 'red', width = 1, label = 'true values')
plt.bar(list(range(len(Y_pred[::print_every]))), Y_pred[::print_every],
        alpha = 0.5, color = 'blue', width = 1, label = 'predicted values')
plt.legend()
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv',parse_dates = ['pickup_datetime'])
key = test['key']
#Data
pdt = test['pickup_datetime']
pdt = pdt.map(lambda date: date.tz_localize(None))
test.drop(['key'], axis = 1, inplace=True)
test['pickup_datetime'] = pdt

#New attributes
test['day_of_week'] = test['pickup_datetime'].map(lambda date: date.timetuple().tm_wday)
test['day_of_year'] = test['pickup_datetime'].map(lambda date: date.timetuple().tm_yday)
test['year'] = test['pickup_datetime'].map(lambda date: date.timetuple().tm_year)
test['hour_of_day'] = test['pickup_datetime'].map(lambda date: date.timetuple().tm_hour)
test.drop('pickup_datetime', axis = 1, inplace = True)

#Distance
lon1, lon2 = np.radians(test['pickup_longitude']), np.radians(test['dropoff_longitude'])
lat1, lat2 = np.radians(test['pickup_latitude']), np.radians(test['dropoff_latitude'])
dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
test['eucl_distance'] = 6373 * c

a1 = np.sin(dlon/2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1-a1))
a2 = np.sin(dlat/2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1-a2))
test['manh_distance'] = 6373 * (c1+c2)
X_test = test
X_test_scaled = scaler.transform(X_test) #scaled between 0 and 1
Y_pred = model.predict(X_test)
sub = pd.DataFrame({'key': key, 'fare_amount': Y_pred})
sub.head()
sub.to_csv('submission.csv', index = False)