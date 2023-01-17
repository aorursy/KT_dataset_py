# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

from scipy import stats

from scipy.stats import norm, skew

from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, plot_importance

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import ElasticNet
df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1000000)

df.head()
df.shape
df.isnull().sum().sort_index()/len(df)
df.describe()
df.dropna(subset=['dropoff_latitude', 'dropoff_longitude'], inplace = True)
df.drop(df[df['fare_amount'] < 0].index, axis=0, inplace = True)
df[df['passenger_count'] > 5].sort_values('passenger_count')
df.drop(df[df['pickup_longitude'] == 0].index, axis=0, inplace = True)

df.drop(df[df['pickup_latitude'] == 0].index, axis=0, inplace = True)

df.drop(df[df['dropoff_longitude'] == 0].index, axis=0, inplace = True)

df.drop(df[df['dropoff_latitude'] == 0].index, axis=0, inplace = True)

df.drop(df[df['passenger_count'] == 208].index, axis=0, inplace = True)
df[df['passenger_count'] > 5].sort_values('passenger_count')
df[df['passenger_count'] > 6].sort_values('passenger_count')
df['key'] = pd.to_datetime(df['key'])

df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
df['Year'] = df['pickup_datetime'].dt.year

df['Month'] = df['pickup_datetime'].dt.month

df['Date'] = df['pickup_datetime'].dt.day

df['Day of Week'] = df['pickup_datetime'].dt.dayofweek

df['Hour'] = df['pickup_datetime'].dt.hour

df.drop('pickup_datetime', axis = 1, inplace = True)

df.drop('key', axis = 1, inplace = True)
df.describe()
df.drop(df[df['pickup_longitude'] < -180].index , inplace=True)

df.drop(df[df['pickup_longitude'] > 180].index , inplace=True)

df.drop(df[df['pickup_latitude'] < -90].index , inplace=True)

df.drop(df[df['pickup_latitude'] > 90].index , inplace=True)



df.drop(df[df['dropoff_longitude'] < -180].index , inplace=True)

df.drop(df[df['dropoff_longitude'] > 180].index , inplace=True)

df.drop(df[df['dropoff_latitude'] < -90].index , inplace=True)

df.drop(df[df['dropoff_latitude'] > 90].index , inplace=True)
df.describe()
import geopy.distance



def calc_miles(trip):

    pickup_lat = trip['pickup_latitude']

    pickup_long = trip['pickup_longitude']

    dropoff_lat = trip['dropoff_latitude']

    dropoff_long = trip['dropoff_longitude']

    distance = geopy.distance.geodesic((pickup_lat, pickup_long), 

                                       (dropoff_lat, dropoff_long)).miles

    try:

        return distance

    except ValueError:

        return np.nan
df['miles'] = df.apply(lambda x: calc_miles(x), axis = 1 )
df.describe()
df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
plt.figure(figsize=(10, 8))

sns.heatmap(df.drop('fare_amount', axis=1).corr(), square=True)

plt.suptitle('Pearson Correlation Heatmap')

plt.show();
(mu, sigma) = norm.fit(df['miles'])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))

ax1 = sns.distplot(df['miles'] , fit=norm, ax=ax1)

ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')

ax1.set_ylabel('Frequency')

ax1.set_title('Miles Distribution')

ax2 = stats.probplot(df['miles'], plot=plt)

f.show();
df[df['miles'] > 100].sort_values('passenger_count')
df.drop(df[df['miles'] > 100].index, axis=0, inplace = True)
(mu, sigma) = norm.fit(np.log1p(df['fare_amount']))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))

ax1 = sns.distplot(np.log1p(df['fare_amount']) , fit=norm, ax=ax1)

ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')

ax1.set_ylabel('Frequency')

ax1.set_title('Log(1+Fare) Distribution')

ax2 = stats.probplot(np.log1p(df['fare_amount']), plot=plt)

f.show();
X, y = df.drop('fare_amount', axis = 1), df['fare_amount']

y = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)

knn_model.fit(X_train, y_train)

y_train_pred = knn_model.predict(X_train)

y_pred = knn_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
train_results, test_results = [], []

n_estimators_test = [16,32,64,128,256,512,1024]

for i in range(len(n_estimators_test)):

    rf_model = RandomForestRegressor(n_estimators=n_estimators_test[i], max_depth=6, max_features=0.5, n_jobs=-1, oob_score=False)

    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)

    y_pred = rf_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    train_results.append(train_rmse)

    test_results.append(test_rmse)

line_trn = plt.plot(n_estimators_test, train_results, 'r')

line_test = plt.plot(n_estimators_test, test_results, 'b')

plt.ylabel('Accuracy score')

plt.xlabel('# Estimators')

plt.show()
kf = KFold(n_splits=5)

rf_model = RandomForestRegressor(n_estimators=512, max_depth=6, max_features=0.5, n_jobs=-1, oob_score=True)

for train_index, test_index in kf.split(X):

    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]

    y_tr, y_te = y.iloc[train_index], y.iloc[test_index]

    y_tr = np.log1p(y_tr)

    y_te = np.log1p(y_te)

    rf_model.fit(X_tr, y_tr)

    y_train_pred = rf_model.predict(X_tr)

    y_pred = rf_model.predict(X_te)

    print('Train r2 score: ', r2_score(y_train_pred, y_tr))

    print('Test r2 score: ', r2_score(y_te, y_pred))

    train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_tr))

    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    print(f'Train RMSE: {train_rmse:.4f}')

    print(f'Test RMSE: {test_rmse:.4f}\n')
rf_model = RandomForestRegressor(n_estimators=512, max_depth=6, max_features=0.5, n_jobs=-1, oob_score=True)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)

y_pred = rf_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

key = test_df.key
test_df['key'] = pd.to_datetime(test_df['key'])

test_df['pickup_datetime']  = pd.to_datetime(test_df['pickup_datetime'])
test_df['Year'] = test_df['pickup_datetime'].dt.year

test_df['Month'] = test_df['pickup_datetime'].dt.month

test_df['Date'] = test_df['pickup_datetime'].dt.day

test_df['Day of Week'] = test_df['pickup_datetime'].dt.dayofweek

test_df['Hour'] = test_df['pickup_datetime'].dt.hour

test_df.drop('pickup_datetime', axis = 1, inplace = True)

test_df.drop('key', axis = 1, inplace = True)

test_df['miles'] = test_df.apply(lambda x: calc_miles(x), axis = 1 )

test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
test_preds = rf_model.predict(test_df)
submission = pd.DataFrame(

    {'key': key, 'fare_amount': np.expm1(test_preds)},

    columns = ['key', 'fare_amount'])

submission.to_csv('rf_submission.csv', index = False)
test_preds_knn = knn_model.predict(test_df)

submission = pd.DataFrame(

    {'key': key, 'fare_amount': np.expm1(test_preds_knn)},

    columns = ['key', 'fare_amount'])

submission.to_csv('knn_submission.csv', index = False)
xgb_model = XGBRegressor(n_estimators=512, colsample_bytree = 0.5, learning_rate = 0.1,

                         max_depth = 10, alpha = 10, n_jobs=-1)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5,

               eval_set=[(X_test, y_test)], verbose=False)

y_train_pred = xgb_model.predict(X_train)

y_pred = xgb_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
train_results, test_results = [], []

n_estimators_test = [16,32,64,128,256,512,1024]

for i in range(len(n_estimators_test)):

    xgb_model = XGBRegressor(n_estimators=n_estimators_test[i], colsample_bytree = 0.5, learning_rate = 0.05,

                             max_depth = 10, alpha = 10, n_jobs=-1)

    xgb_model.fit(X_train, y_train, early_stopping_rounds=5,

                  eval_set=[(X_test, y_test)], verbose=False)

    y_train_pred = xgb_model.predict(X_train)

    y_pred = xgb_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    train_results.append(train_rmse)

    test_results.append(test_rmse)

line_trn = plt.plot(n_estimators_test, train_results, 'r')

line_test = plt.plot(n_estimators_test, test_results, 'b')

plt.ylabel('Accuracy score')

plt.xlabel('# Estimators')

plt.show()
kf = KFold(n_splits=5)

xgb_modelkf = XGBRegressor(n_estimators=200, colsample_bytree = 0.5, learning_rate = 0.05,

                           max_depth = 7, alpha = 10, n_jobs=-1)

for train_index, test_index in kf.split(X):

    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]

    y_tr, y_te = y.iloc[train_index], y.iloc[test_index]

    y_tr = np.log1p(y_tr)

    y_te = np.log1p(y_te)

    xgb_modelkf.fit(X_tr, y_tr, early_stopping_rounds=5,

                    eval_set=[(X_te, y_te)], verbose=False)

    y_train_pred = xgb_modelkf.predict(X_tr)

    y_pred = xgb_modelkf.predict(X_te)

    print('Train r2 score: ', r2_score(y_train_pred, y_tr))

    print('Test r2 score: ', r2_score(y_te, y_pred))

    train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_tr))

    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    print(f'Train RMSE: {train_rmse:.4f}')

    print(f'Test RMSE: {test_rmse:.4f}\n')
xgb_model = XGBRegressor(n_estimators=200, colsample_bytree = 0.5, learning_rate = 0.05,

                         max_depth = 10, alpha = 10, n_jobs=-1)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5,

              eval_set=[(X_test, y_test)], verbose=False)

y_train_pred = xgb_model.predict(X_train)

y_pred = xgb_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
test_preds_xgb = xgb_model.predict(test_df)

submission = pd.DataFrame(

    {'key': key, 'fare_amount': np.expm1(test_preds_xgb)},

    columns = ['key', 'fare_amount'])

submission.to_csv('xgb_submission.csv', index = False)