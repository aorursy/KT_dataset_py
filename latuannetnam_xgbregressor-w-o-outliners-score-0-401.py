# data processing

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm

import math

from numpy import sort

from math import radians, cos, sin, asin, sqrt

# ML

# # Scikit-learn

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import SelectFromModel

# # XGB

from xgboost.sklearn import XGBRegressor

import xgboost as xgb

from xgboost import plot_importance

# # CatBoost

#from catboost import Pool, CatBoostRegressor, cv, CatboostIpythonWidget

# System

import datetime as dtime

from datetime import datetime

import sys

from inspect import getsourcefile

import os.path

import re

import time



# Other

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Load data. Download from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data

# Input data files are available in the DATA_DIR directory.

DATA_DIR = "../input"

train_data = pd.read_csv(DATA_DIR + "/train.csv")

eval_data =  pd.read_csv(DATA_DIR + "/test.csv")
print("train size:", train_data.shape, " test size:", eval_data.shape)
train_data.head(5)
eval_data.head(5)
diff_cols = np.setdiff1d(train_data.columns.values, eval_data.columns.values)

diff_cols
label = 'trip_duration'

features = eval_data.columns.values

target = train_data[label].values

combine_data = pd.concat([train_data[features], eval_data], keys=['train','eval'])

combine_data.head(5)
def check_null_data(data):

    #Get high percent of NaN data

    null_data = combine_data.isnull()

    total = null_data.sum().sort_values(ascending=False)

    percent = (null_data.sum()/null_data.count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    high_percent_miss_data = missing_data[missing_data['Percent']>0]

    #print(missing_data)

    print(high_percent_miss_data)

    miss_data_cols = high_percent_miss_data.index.values

    return miss_data_cols
# Check target for null

check_null_data(target)
# combine data for null

check_null_data(combine_data)
combine_data.dtypes
data = combine_data
#data['datetime_obj'] = pd_datetime(data)

data['datetime_obj'] = pd.to_datetime(data['pickup_datetime'])

data['datetime_obj'][:5]
data['pickup_year'] = data['datetime_obj'].dt.year

data['pickup_month'] = data['datetime_obj'].dt.month

data['pickup_weekday'] = data['datetime_obj'].dt.weekday

data['pickup_day'] = data['datetime_obj'].dt.day

data['pickup_hour'] = data['datetime_obj'].dt.hour

data['pickup_minute'] = data['datetime_obj'].dt.minute
data[:5]
col = 'store_and_fwd_flag'

data[col].value_counts()
data_dict = {'Y':1, 'N':0}

data_tf = data[col].map(data_dict)

data[col].update(data_tf)

data[:5]
# Drop pickup_datetime

data.drop('pickup_datetime', axis=1, inplace=True)

data.drop('datetime_obj', axis=1, inplace=True)
combine_data_tf = data
combine_data_tf['pickup_year'].value_counts()
# Drop pickup year

combine_data_tf.drop('pickup_year', axis=1, inplace=True)
# credit to: https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points

def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    km = 6367 * c

    return km
data = combine_data_tf

data['distance'] = combine_data_tf.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

combine_data_tf[:5]
data = combine_data_tf

train_set = data.loc['train']

eval_set = data.loc['eval']

data = train_set

data[label] = target

target_log = np.log(target)

data[:5]
plt.scatter(data.index, data[label])
data = train_set

data_ol = data[data[label] > 1800000]

data_ol[-5:]
data_ol = data[data[label] < 1800000]

plt.scatter(data_ol.index, data_ol[label])
train_set = data_ol

data = train_set

# use np.log to balance distribution

target_log = np.log(train_set[label].values)
train_set[label].describe()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(data[label], bins=50)

plt.title('Original Data')

plt.xlabel(label)

plt.subplot(1,2,2)

sns.distplot(target_log, bins=50)

plt.title('Natural Log of Data')

plt.xlabel('Natural Log of ' + label)

correlation = data.corr()[label].sort_values()[-20:]

correlation
#correlation matrix

corrmat = data.corr()

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, label)[label].index

print(cols.values)

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Correlation matrix of cols except label

other_cols = np.setdiff1d(cols.values, label)

cm = np.corrcoef(data[other_cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=other_cols, xticklabels=other_cols)

plt.show()
data = train_set.drop(['id', label], axis=1).astype(float)

print(data.shape)

data[:5]
# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

# credit: https://www.kaggle.com/marknagelberg/rmsle-function





def rmsle(y, y_pred, log=True):

    assert len(y) == len(y_pred)

    terms_to_sum = 0

    if log:

        terms_to_sum = [(math.log(math.fabs(y_pred[i]) + 1) -

                         math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]

    else:

        terms_to_sum = [(math.fabs(y_pred[i]) - y[i]) **

                        2.0 for i, pred in enumerate(y_pred)]

    # for i, pred in enumerate(y_pred):

    #    print("i:", i, " pred:", y_pred[i])

    #    print(math.log(y_pred[i] + 1))

    #    print(math.log(y[i] + 1))

    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5
# set n_estimator=5000  to increase score

model = XGBRegressor(n_estimators=10, max_depth=5,

                     learning_rate=0.1, min_child_weight=1, n_jobs=-1)
X_train, X_test, Y_train, Y_test = train_test_split(

    data, target_log, train_size=0.85, random_state=1234)

print("X_train:", X_train.shape, " Y_train:", Y_train.shape,

      " X_test:", X_test.shape, " Y_test:", Y_test.shape)

X_train[:5]
start = time.time()

early_stopping_rounds = 50

model.fit(

    X_train, Y_train, eval_set = [(X_test, Y_test)],

    eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,

    verbose=early_stopping_rounds

)

end = time.time() - start

print(end)
start = time.time()

y_pred = model.predict(X_test)

end = time.time() - start

end
# Evaluate score

print(y_pred[:5])

score = rmsle(Y_test, y_pred)

score1 = rmsle(Y_test, y_pred, log=False)

print("RMSLE score:", score, " RMSLE without-log:", score1)
plot_importance(model)

plt.show()
start = time.time()

data = eval_set.drop('id', axis=1).astype(float)

Y_eval_log = model.predict(data)

Y_eval = np.exp(Y_eval_log.ravel())

end = time.time() - start

print(end)

print(Y_eval_log[:5])

print(Y_eval[:5])
eval_output = pd.DataFrame({'id': eval_data['id'], 'trip_duration': Y_eval})

print(len(eval_output))

eval_output.head()
start = time.time()

today = str(dtime.date.today())

print(today)

eval_output.to_csv(today+'-submission.csv',index=False)

end = time.time() - start

end