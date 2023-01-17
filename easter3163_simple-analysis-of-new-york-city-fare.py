import pandas as pd
train = pd.read_csv('../input/train.csv', nrows=300_000)
test = pd.read_csv('../input/test.csv')
train.shape
train.head()
%matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['hour'] = train['pickup_datetime'].dt.hour
train['day'] = train['pickup_datetime'].dt.day
train['week'] = train['pickup_datetime'].dt.week
train['month'] = train['pickup_datetime'].dt.month
train['day_of_year'] = train['pickup_datetime'].dt.dayofyear
train['week_of_year'] = train['pickup_datetime'].dt.weekofyear
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['hour'] = test['pickup_datetime'].dt.hour
test['day'] = test['pickup_datetime'].dt.day
test['week'] = test['pickup_datetime'].dt.week
test['month'] = test['pickup_datetime'].dt.month
test['day_of_year'] = test['pickup_datetime'].dt.dayofyear
test['week_of_year'] = test['pickup_datetime'].dt.weekofyear
train.head()
train = train.dropna(how = 'any', axis='rows')
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -75) & (train['pickup_longitude'] < 75)]
train = train.loc[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 45)]
train = train.loc[(train['dropoff_longitude'] > -75) & (train['dropoff_longitude'] < 75)]
train = train.loc[(train['dropoff_latitude'] > 40) & (train['dropoff_latitude'] < 45)]
train = train.loc[train['passenger_count'] <= 8]
train['abs_diff_longitude'] = (train['pickup_longitude'] - train['dropoff_longitude']).abs()
train['abs_diff_latitude'] = (train['pickup_latitude'] - train['dropoff_latitude']).abs()
test['abs_diff_longitude'] = (test['pickup_longitude'] - test['dropoff_longitude']).abs()
test['abs_diff_latitude'] = (test['pickup_latitude'] - test['dropoff_latitude']).abs()
train.head()
train.head()
sns.barplot(data=train, x="passenger_count", y="fare_amount")
#'hour', 'passenger_count'
feature_names = ['hour', 'passenger_count','abs_diff_longitude', 'abs_diff_latitude']
feature_names
label_name = 'fare_amount'
label_name
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb
#Linear Regression Model
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)
#KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)
#Random Forest Model
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
#set parameters for xgboost
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50
xb = xgb.train(params, dtrain, num_rounds)
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)
#Assigning weights
# predictions = (regr_prediction + rfr_prediction + knr_prediction + 3 * y_pred_xgb) / 6
predictions = y_pred_xgb
predictions
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = predictions
submission.head()
submission.to_csv('./simplenewyorktaxi.csv', index=False)