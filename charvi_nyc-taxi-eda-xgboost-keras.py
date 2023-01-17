# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import io

import gc

import xgboost as xgb

import keras

import sklearn

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/new-york-city-taxi-with-osrm/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
#data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype(object)

data.loc[data.store_and_fwd_flag=='N','store_and_fwd_flag'] = 0

data.loc[data.store_and_fwd_flag=='Y','store_and_fwd_flag'] = 1

data.loc[data.vendor_id==2,'vendor_id'] = 0

data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype(int)

data['store_and_fwd_flag'] = data['store_and_fwd_flag'].fillna(0)



#test['store_and_fwd_flag'] = test['store_and_fwd_flag'].astype(object)

test.loc[test.store_and_fwd_flag=='N','store_and_fwd_flag'] = 0

test.loc[test.store_and_fwd_flag=='Y','store_and_fwd_flag'] = 1

test.loc[test.vendor_id==2,'vendor_id'] = 0

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].astype(int)

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].fillna(0)
data['distance'] = abs(data.dropoff_longitude - data.pickup_longitude) + abs(data.dropoff_latitude - data.dropoff_latitude)

test['distance'] = abs(test.dropoff_longitude - test.pickup_longitude) + abs(test.dropoff_latitude - test.dropoff_latitude)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

data.describe()
corr = data.corr('spearman')

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr, vmax=1., square=True, cmap='Greens')

plt.title("Feature correlation map", fontsize=15)

plt.show()
sns.set_style('darkgrid')

fig, ax = plt.subplots(figsize=(9,7))

# < 1000 data points have distance > 0.3

sns.distplot(data[data.distance<0.3]['distance'],bins = 100)
data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)

data.loc[:, 'pickup_dayofweek'] = data['pickup_datetime'].dt.dayofweek

data.loc[:, 'pickup_month'] = data['pickup_datetime'].dt.month

data.loc[:, 'pickup_hour'] = data['pickup_datetime'].dt.hour

data.loc[:, 'pickup_hour_weekofyear'] = data['pickup_datetime'].dt.weekofyear

data.loc[:, 'pickup_minute'] = data['pickup_datetime'].dt.minute

data.loc[:, 'pickup_week_hour'] = data['pickup_dayofweek'] * 24 + data['pickup_hour']



test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

test.loc[:, 'pickup_dayofweek'] = test['pickup_datetime'].dt.dayofweek

test.loc[:, 'pickup_month'] = test['pickup_datetime'].dt.month

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_week_hour'] = test['pickup_dayofweek'] * 24 + test['pickup_hour']
data.loc[:, 'speed'] = 1000 * data['distance'] / data['trip_duration']

sns.set_style(style='darkgrid')

fig, ax = plt.subplots(ncols=3, sharey=True,figsize=(12,6))

ax[0].plot(data.groupby('pickup_hour').mean()['speed'], 'bo-', lw=2, alpha=0.7)

ax[1].plot(data.groupby('pickup_dayofweek').mean()['speed'], 'go-', lw=2, alpha=0.7)

ax[2].plot(data.groupby('pickup_month').mean()['speed'], 'ro-', lw=2, alpha=0.7)

ax[0].set_xlabel('Hour of Day')

ax[1].set_xlabel('Day of Week')

ax[2].set_xlabel('Month of Year')

ax[0].set_ylabel('Average Speed')

fig.suptitle('Average Traffic Speed by Date/Time')

plt.show()
osrm1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv',usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])

osrm2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv',usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])

test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',

                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_street_info = pd.concat((osrm1, osrm2))

data = data.merge(train_street_info, how='left', on='id')

test = test.merge(test_street_info, how='left', on='id')
# from beluga's notebook

coords = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values,

                    data[['dropoff_latitude', 'dropoff_longitude']].values,

                    test[['pickup_latitude', 'pickup_longitude']].values,

                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)

data['pickup_pca_0'] = pca.transform(data[['pickup_latitude', 'pickup_longitude']])[:, 0]

data['pickup_pca_1'] = pca.transform(data[['pickup_latitude', 'pickup_longitude']])[:, 1]

data['dropoff_pca_0'] = pca.transform(data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

data['dropoff_pca_1'] = pca.transform(data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca_0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca_1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca_0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca_1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
corr = data.corr('spearman')

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr, vmax=1., square=True, cmap='Blues')

plt.title("Final feature correlation map", fontsize=15)

plt.show()
gc.collect()
train_cols = [col for col in data.columns if col not in ['speed','id', 'log_trip_duration', \

                                                         'pickup_datetime', 'dropoff_datetime',\

                                                         'trip_duration']]



x_full = data[train_cols]

y_full = y = np.log(data['trip_duration'].values + 1)



x_train, x_test, y_train, y_test = train_test_split(x_full,y_full, test_size=0.3)
dtrain = xgb.DMatrix(x_train, label=y_train)

params = {'min_child_weight': 50, 'eta': 0.5, 'colsample_bytree': 0.3, 'max_depth': 20,

            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear','gamma' : 1}

dvalid = xgb.DMatrix(x_test, label=y_test)

dtest = xgb.DMatrix(test[train_cols])

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



model  = xgb.train(params,dtrain,80, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=10)

'''

[69]	train-rmse:0.296599	valid-rmse:0.385196

params = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 20,

            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear','gamma' : 1}

'''
ytest = model.predict(dtest)

test['trip_duration'] = np.exp(ytest) - 1

test[['id','trip_duration']].to_csv("kaggle_nyc_submission.csv",index=False)
# test with keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
def larger_model():

    # create model

    model = Sequential()

    model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))

    model.add(Dense(11, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

train_cols = [col for col in data.columns if col not in ['speed','id', 'log_trip_duration', \

                                                         'pickup_datetime', 'dropoff_datetime',\

                                                         'trip_duration']]

keras_data = data.dropna()

x_full = keras_data[train_cols]

y_full = np.log(keras_data['trip_duration'].values + 1)



x_train, x_test, y_train, y_test = train_test_split(x_full,y_full, test_size=0.3)
nn = larger_model()

nn.fit( x_full, y_full, batch_size=len(x_full), epochs=100, verbose=1)

ytest = nn.predict(test[train_cols],batch_size=len(test))

test['trip_duration'] = np.exp(ytest) - 1

test[['id','trip_duration']].to_csv("kaggle_nyc_submission_keras.csv",index=False)