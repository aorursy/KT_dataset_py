import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt
import gc
import time
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import matplotlib
import matplotlib.pyplot as plt
% matplotlib inline
plt.rcParams['figure.figsize'] = [15, 6]
plt.style.use('ggplot')
train_df = pd.read_csv("../input/new-york-city-taxi-with-osrm/train.csv")
test_df = pd.read_csv("../input/new-york-city-taxi-with-osrm/test.csv")
# Read OSRM csv data
fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_osrm_df = pd.concat((fr1,fr2))
test_osrm_df = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_df = pd.merge(train_df, train_osrm_df, on='id', how='left').fillna(0)
test_df = pd.merge(test_df, test_osrm_df, on='id', how='left').fillna(0)
# columns type
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["dropoff_datetime"] = pd.to_datetime(train_df["dropoff_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["vendor_id"] = train_df["vendor_id"].astype("str")
test_df["vendor_id"] = test_df["vendor_id"].astype("str")

le = preprocessing.LabelEncoder()
train_df["store_and_fwd_flag"] = le.fit_transform(train_df["store_and_fwd_flag"])
test_df["store_and_fwd_flag"] = le.fit_transform(test_df["store_and_fwd_flag"])
# The evaluation metric for this competition is Root Mean Squared Logarithmic Error.
def root_mean_squeared_log_error(y_pred, y_true):
    error = np.sqrt(np.sum((np.log(y_pred + 1) - np.log(y_true + 1))**2)/len(y_pred))
    return error
target_col = "trip_duration"
exclude_feature_cols = ["id", "pickup_datetime", "dropoff_datetime", "trip_duration"]
feature_cols = [col for col in train_df.columns if col not in exclude_feature_cols]

y_train = train_df[target_col].values
X_train = train_df[feature_cols].values
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.3, random_state = 1234)

X_test = test_df[feature_cols].values
# Cross Validation (Random Forest)

# params = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 50]}
# rf = RandomForestRegressor(random_state=1234)

# my_scorer = make_scorer(root_mean_squeared_log_error, greater_is_better=False)
# gscv = GridSearchCV(rf, param_grid=params, verbose=1, cv=3, scoring=my_scorer)

# start = time.time()
# gscv.fit(X_train, y_train)
# print(time.time() - start)
# Train the data
rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=1234)
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
rf_rmlse_train = root_mean_squeared_log_error(y_pred_train, y_train)
print("RF Train-data RMLSE", rf_rmlse_train)
# Validate the data
y_pred_validate = rf.predict(X_validate)
rf_rmlse_validate = root_mean_squeared_log_error(y_pred_validate, y_validate)
print("RF Valdation-data RMLSE", rf_rmlse_validate)
# Apply to the Test data
y_pred = rf.predict(X_test)
submission_df = pd.DataFrame({'id':test_df["id"], 'trip_duration': y_pred})
submission_df.to_csv('sub_rf_v0.csv',index=False)  # 0.57367
pd.set_option('display.float_format', lambda x: '%.3f' % x)
train_df.describe()
# deal with the outlier of target variable
mean = np.mean(train_df["trip_duration"])
sd = np.std(train_df["trip_duration"])

train = train_df[train_df["trip_duration"] >= mean - 3*sd]
train = train_df[train_df["trip_duration"] <= mean + 3*sd]
test = test_df
plt.hist(train["trip_duration"], bins=100)
plt.show()
plt.hist(np.log(train["trip_duration"]), bins=100)
plt.show()
# make log data for trip_duration
train["trip_duration_log"] = np.log(train["trip_duration"])
# vendor_id
train[["vendor_id", "trip_duration_log"]].boxplot(by="vendor_id")
# passenger_count
train[["passenger_count", "trip_duration_log"]].boxplot(by="passenger_count")
# store_and_fwd_flag
train[["store_and_fwd_flag", "trip_duration_log"]].boxplot(by="store_and_fwd_flag")
# day to category 
train["Month"] = train["pickup_datetime"].dt.month
test["Month"] = test["pickup_datetime"].dt.month
print(sorted(train["Month"].unique()))
print(sorted(test["Month"].unique()))

train["Day"] = train["pickup_datetime"].dt.day
test["Day"] = test["pickup_datetime"].dt.day
print(sorted(train["Day"].unique()))
print(sorted(test["Day"].unique()))

train["DayofWeek"] = train["pickup_datetime"].dt.dayofweek
test["DayofWeek"] = test["pickup_datetime"].dt.dayofweek
print(sorted(train["DayofWeek"].unique()))
print(sorted(test["DayofWeek"].unique()))

train["Hour"] = train["pickup_datetime"].dt.hour
test["Hour"] = test["pickup_datetime"].dt.hour
print(sorted(train["Hour"].unique()))
print(sorted(test["Hour"].unique()))
# day
train[["Month", "trip_duration_log"]].boxplot(by="Month")
train[["Day", "trip_duration_log"]].boxplot(by="Day")
train[["DayofWeek", "trip_duration_log"]].boxplot(by="DayofWeek")
train[["Hour", "trip_duration_log"]].boxplot(by="Hour")
# total_distance
plt.scatter(train["total_distance"],train["trip_duration_log"])
plt.scatter(np.log(train["total_distance"]),train["trip_duration_log"])
# # make log data for total_distance
# train["total_distance_log"] = np.log(train["total_distance"])
# test["total_distance_log"] = np.log(test["total_distance"])
plt.scatter(train["number_of_steps"],train["trip_duration_log"])
# make dummy variables
vendor_train = pd.get_dummies(train["vendor_id"],  prefix='vendor')
passenger_count_train = pd.get_dummies(train["passenger_count"],  prefix='pc_cnt')
store_and_fwd_flag_train = pd.get_dummies(train["store_and_fwd_flag"],  prefix='sf_flg')
month_train = pd.get_dummies(train["Month"],  prefix='month')
day_train = pd.get_dummies(train["Day"],  prefix='day')
dayofweek_train = pd.get_dummies(train["DayofWeek"],  prefix='dow')
hour_train = pd.get_dummies(train["Hour"],  prefix='hour')

train = pd.concat([train, vendor_train, passenger_count_train, store_and_fwd_flag_train,
                  month_train, day_train, dayofweek_train, hour_train], axis=1)

vendor_test = pd.get_dummies(test["vendor_id"],  prefix='vendor')
passenger_count_test = pd.get_dummies(test["passenger_count"],  prefix='pc_cnt')
store_and_fwd_flag_test = pd.get_dummies(test["store_and_fwd_flag"],  prefix='sf_flg')
month_test = pd.get_dummies(test["Month"],  prefix='month')
day_test = pd.get_dummies(test["Day"],  prefix='day')
dayofweek_test = pd.get_dummies(test["DayofWeek"],  prefix='dow')
hour_test = pd.get_dummies(test["Hour"],  prefix='hour')

test = pd.concat([test, vendor_test, passenger_count_test, store_and_fwd_flag_test,
                  month_test, day_test, dayofweek_test, hour_test], axis=1)
print(sorted(train["passenger_count"].unique()))
print(sorted(test["passenger_count"].unique()))

train = train.drop(["pc_cnt_7", "pc_cnt_8"], axis = 1)
print(train.shape)
train.head()
print(test.shape)
test.head()
target_col = "trip_duration_log"
exclude_feature_cols = ["id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count", "store_and_fwd_flag", 
                        "Month", "Day", "DayofWeek", "Hour", "trip_duration", "trip_duration_log"]
feature_cols = [col for col in train.columns if col not in exclude_feature_cols]

y_train_log = train[target_col].values
X_train = train[feature_cols].values
X_train, X_valid, y_train_log, y_valid_log = train_test_split(X_train, y_train_log, test_size = 0.3, random_state = 1234)

X_test = test[feature_cols].values
# Train the data
rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=1234)
rf.fit(X_train, y_train_log)
y_pred_train = rf.predict(X_train)
rf_rmlse_train = root_mean_squeared_log_error(np.exp(y_pred_train), np.exp(y_train_log))
print("RF Train-data RMLSE", rf_rmlse_train)
# Validate the data
y_pred_valid = rf.predict(X_valid)
rf_rmlse_valid = root_mean_squeared_log_error(np.exp(y_pred_valid), np.exp(y_valid_log))
print("RF Valdation-data RMLSE", rf_rmlse_valid)
# Apply to the Test data
y_pred = rf.predict(X_test)
submission_df = pd.DataFrame({'id':test["id"], 'trip_duration': np.exp(y_pred)})
submission_df.to_csv('sub_rf_v2.csv',index=False)  # 0.45818
left = list(range(len(feature_cols)))
height = np.array(rf.feature_importances_)

plt.figure(figsize=(15,25))
plt.barh(left, height, tick_label = feature_cols)
dtrain = xgb.DMatrix(X_train, label=y_train_log)
dvalid = xgb.DMatrix(X_valid, label=y_valid_log)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 1, 'eta': 0.3, 'colsample_bytree': 0.9,  'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}

xgb_reg = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
# Apply to the Test data
y_pred = xgb_reg.predict(dtest)
submission_df = pd.DataFrame({'id':test["id"], 'trip_duration': np.exp(y_pred)})
submission_df.to_csv('sub_xgb_v1.csv',index=False)  # 0.48662