import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy.stats import skew, norm

from sklearn.cluster import KMeans

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/train.zip')

test  = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/test.zip')

sub   = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/sample_submission.zip')
# target 인 trip duration과 target을 계산 할 수 있는 dropoff time 이 없다. 

print(train.shape)

print(test.shape)
train.info()
train.head()
def datetime_split(data) :

    data['pickup_datetime'] = data.pickup_datetime.apply(pd.to_datetime)

    data['year'] = data.pickup_datetime.apply(lambda x : x.year)

    data['month'] = data.pickup_datetime.apply(lambda x : x.month)

    data['day'] = data.pickup_datetime.apply(lambda x : x.day)

    data['hour'] = data.pickup_datetime.apply(lambda x  : x.hour)

    data['dayofweek'] = data.pickup_datetime.apply(lambda x : x.dayofweek)

    

datetime_split(train)

datetime_split(test)
train.head()
train.year.value_counts() 



train.drop('year', axis = 1,  inplace = True)

test.drop('year', axis = 1, inplace = True)
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



train['haversine'] = haversine_array(train['pickup_latitude'], 

                                     train['pickup_longitude'],

                                     train['dropoff_latitude'],

                                     train['dropoff_longitude'])



train['manhattan'] = dummy_manhattan_distance(train['pickup_latitude'], 

                                     train['pickup_longitude'],

                                     train['dropoff_latitude'],

                                     train['dropoff_longitude'])

test['haversine'] = haversine_array(test['pickup_latitude'], 

                                     test['pickup_longitude'],

                                     test['dropoff_latitude'],

                                     test['dropoff_longitude'])



test['manhattan'] = dummy_manhattan_distance(test['pickup_latitude'], 

                                     test['pickup_longitude'],

                                     test['dropoff_latitude'],

                                     test['dropoff_longitude'])

train.head()
xlim = [-74.03, -73.77] 

ylim = [40.63, 40.85]
plt.scatter(train['pickup_longitude'].values[:100000], train['pickup_latitude'].values[:100000],

              color='blue', s=1, label='train', alpha=0.1)

plt.xlim(xlim)

plt.ylim(ylim)
loc_df = pd.DataFrame()

longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)

latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)

loc_df['longitude'] = longitude

loc_df['latitude'] = latitude





kmeans = KMeans(n_clusters=12, random_state=2, n_init = 20).fit(loc_df)



loc_df['label'] = kmeans.labels_



loc_df = loc_df.sample(200000)

plt.figure(figsize = (10,10))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize = 0.3)



plt.xlim(xlim)

plt.ylim(ylim)

plt.title('Clusters of New York')

plt.show()
target = train['trip_duration']



sns.distplot(target, fit = norm)



plt.xlabel('Trip_duration')
sns.boxplot(target)
train[train['trip_duration'] > 1500000]
train.drop(train[train['trip_duration'] > 150000].index, axis = 0, inplace = True)
sns.distplot(train['trip_duration'], fit = norm)



plt.xlabel('Trip_duration')
train[train['trip_duration'] >= 20000]
Q1 = np.percentile(train['trip_duration'], 25) 

Q3 = np.percentile(train['trip_duration'], 75)

IQR = Q3 - Q1 

outlier_step = 1.5 * IQR 

outlier_list_idx = train[train['trip_duration'] > Q3 + outlier_step].index



train.drop(outlier_list_idx, axis = 0, inplace = True)
sns.distplot(train['trip_duration'], fit = norm)



plt.title('skewness of target {0:.3f}'.format(skew(train['trip_duration'])))
sns.distplot(np.log1p(train['trip_duration']), fit = norm)



plt.xlabel('trip_duration')

plt.title('skewness of target {0:.3f}'.format(skew(np.log1p(train['trip_duration']))))
ln_target = np.log1p(train['trip_duration'])

train['ln_target'] = ln_target
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)



axes[0].scatter(x = train['haversine'],

                y = train['trip_duration'])

axes[0].set_title('Target ~ haversine distance')

axes[0].set_xlabel('haversine distance')



axes[1].scatter(x = train['manhattan'],

                y = train['trip_duration'], )

axes[1].set_title('Target ~ manhattan distance')

axes[1].set_xlabel('manhattan distance')



plt.tight_layout()

plt.show()
sns.boxplot(test['haversine'])



plt.xlabel('haversine')

plt.title('Boxplot of haversine')
fig, axes = plt.subplots(1, 2, sharex=False)



axes[0].boxplot(train['haversine'])

axes[0].set_title('haversine distance \n skewness {:.3f}'.format(skew(train['haversine'])))

axes[0].set_xlabel('haversine distance')



axes[1].boxplot(np.log1p(train['haversine']))

axes[1].set_title('Log haversine distance \n skewness {:.3f}'.format(skew(np.log1p(train['haversine']))))

axes[1].set_xlabel('log haversine distance')





plt.tight_layout()

plt.show()
snfig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

train['ln_haversine'] = np.log1p(train['haversine'])

train['ln_manhattan'] = np.log1p(train['manhattan'])

test['ln_haversine'] = np.log1p(test['haversine'])

test['ln_manhattan'] = np.log1p(test['manhattan'])



axes[0].scatter(x = train['ln_haversine'],

                y = train['trip_duration'])

axes[0].set_title('Target ~ Log haversine distance \n correlation : {:.3f}'.format(train[['ln_haversine','trip_duration']].corr().iloc[1,0]))

axes[0].set_xlabel('Log haversine distance')



axes[1].scatter(x = train['ln_manhattan'],

                y = train['trip_duration'])

axes[1].set_title('Target ~ Log manhattan distance \n correlation : {:.3f}'.format(train[['ln_manhattan','trip_duration']].corr().iloc[1,0]))

axes[1].set_xlabel('Log manhattan distance')



plt.tight_layout()

plt.show()
snfig, axes = plt.subplots(1, 2, sharex=True, sharey=True)



axes[0].scatter(x = train['ln_haversine'],

                y = train['ln_target'])

axes[0].set_title('Log Target ~ Log haversine distance\n correlation : {:.3f}'.format(train[['ln_haversine','ln_target']].corr().iloc[1,0]))

axes[0].set_xlabel('Log haversine distance')



axes[1].scatter(x = train['ln_manhattan'],

                y = train['ln_target'])

axes[1].set_title('Log Target ~ Log manhattan distance \n correlation : {:.3f}'.format(train[['ln_manhattan','ln_target']].corr().iloc[1,0]))

axes[1].set_xlabel('Log manhattan distance')



plt.tight_layout()

plt.show()
train[['haversine', 'manhattan','ln_haversine','ln_manhattan','trip_duration','ln_target']].corr()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

# linear model 

ln_target = train['ln_target']

lr_m = LinearRegression()

poly_m = LinearRegression()



train_distance = train.ln_haversine.values



train_distance = train_distance.reshape(-1, 1)



poly = PolynomialFeatures(degree=2)



train_dist_sqr = poly.fit_transform(train_distance)



lr_m.fit(train_distance, ln_target)

poly_m.fit(train_dist_sqr, ln_target)



pred_lr = lr_m.predict(train_distance)

pred_poly = poly_m.predict(train_dist_sqr)



lr_score = mean_squared_error(ln_target, pred_lr)

poly_score = mean_squared_error(ln_target, pred_poly)



r2_lr = r2_score(ln_target, pred_lr)

r2_poly = r2_score(ln_target, pred_poly)



print('r2_score (degree = 1) : {0:.3f} \n  MSE : {1:.3f}'.format(r2_lr, lr_score))

print('======================================================')

print('r2_score(degree = 2) : {0:.3f} \n MSE : {1:.3f}'.format(r2_poly, poly_score))

print('======================================================')

print('polynomial regression estimators : ({1:.3f}) * hour^2 + ({0:.3f}) * hour + ({2:.3f})'

      .format(poly_m.coef_[1],poly_m.coef_[2], poly_m.intercept_))
sns.boxplot(x = train['hour'],

            y = train['ln_target'])





plt.title('Log trip duration ~ Hour')
sns.boxplot(x = train['dayofweek'],

            y = train['ln_target'])



plt.title('Log Target ~ Dayofweek')
weekend = pd.DataFrame({'dayofweek' : [0,1,2,3,4,5,6], 'weekend' : [1,0,0,0,0,0,1]})



train = pd.merge(train, weekend, on = 'dayofweek')

test = pd.merge(test, weekend, on = 'dayofweek')

train.head()
sns.boxplot(x = train['weekend'],

            y = train['ln_target'])
train.groupby(by = 'weekend').boxplot(column = ['ln_target'], by = ['hour'], figsize = (20,20))
sns.boxplot(x = train['passenger_count'],

            y = train['ln_target'])
print(train.passenger_count.value_counts())

print('==============================')

print(test.passenger_count.value_counts())
train['passenger_count'].replace(0,1, inplace= True)

train['passenger_count'].replace(7,6, inplace= True)

train['passenger_count'].replace(8,6, inplace= True)

train['passenger_count'].replace(9,6, inplace= True)



test['passenger_count'].replace(0,1, inplace= True)

test['passenger_count'].replace(9,6, inplace= True)

train.passenger_count.value_counts()
sns.boxplot(x = train['passenger_count'],

            y = train['ln_target'])
sns.boxplot(x = train['store_and_fwd_flag'],

            y = train['ln_target'])



plt.title('Log Target ~ store and fwd flag')