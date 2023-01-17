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
# !unzip /kaggle/input/nyc-taxi-trip-duration/test.zip
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

from geopy.distance import great_circle

import numpy as np

import gc

import datetime

import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV

import statsmodels.api as sm

%matplotlib inline

from sklearn.metrics import accuracy_score, mean_squared_log_error

from collections import OrderedDict

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from scipy.stats import uniform, randint
# def distance(lon1, lat1, lon2, lat2):

#     pick_up = (lat1, lon1)

#     drop_off = (lat2, lon2)

#     return great_circle(pick_up, drop_off).miles



# def reduce_mem_usage(df):

#     """ iterate through all the columns of a dataframe and modify the data type

#         to reduce memory usage.        

#     """

#     #start_mem = df.memory_usage().sum() / 1024**2

#     #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



#     for col in df.columns:

#         col_type = df[col].dtype



#         if col_type != object:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)



#     #end_mem = df.memory_usage().sum() / 1024**2

#     #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

#     #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



#     return df
# train_data = pd.read_csv('train.csv')



# train_data = reduce_mem_usage(train_data)

# train_data.info()
# train_data['distance'] = train_data.apply(lambda x: distance(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']) , axis=1)

# train_data.head()
# train_data['dow'] = train_data['pickup_datetime'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())

# train_data['hour'] = train_data['pickup_datetime'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour) 

# train_data.head()
# v99 = train_data['trip_duration'].quantile(0.99)

# train_data_v99 = train_data[train_data['trip_duration']<=v99]
# sns.set(style="ticks", color_codes=True)

# import matplotlib.pyplot as plt

# g = sns.FacetGrid(train_data, col="vendor_id", height=4)

# g = g.map(plt.hist, "trip_duration")
# store_y = train_data[train_data["store_and_fwd_flag"]=="Y"]

# store_n = train_data[train_data["store_and_fwd_flag"]=="N"]

# store_y['trip_duration'].mean(), store_n['trip_duration'].mean()
# store_y_v99 = store_y['trip_duration'].quantile(0.99)

# store_n_v99 = store_n['trip_duration'].quantile(0.99)



# sns.distplot(store_n[store_n['trip_duration']<store_n_v99]['trip_duration'])

# sns.distplot(store_y[store_y['trip_duration']<store_y_v99]['trip_duration'])



# plt.show()
# dist_v99 = train_data['distance'].quantile(0.99)

# train_dist_trip_v99 = train_data_v99[train_data_v99['distance'] < dist_v99] 
# train_dist_trip_v99['distance_1'] = train_dist_trip_v99['distance'].apply(lambda x : round(x * 2) / 2)

# train_dist_trip_v99 = train_dist_trip_v99.groupby(["distance_1"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# train_dist_trip_v99.head()

# ax = sns.scatterplot(x="distance_1", y="trip_dur", data=train_dist_trip_v99)
# train_data_dow_v99 = train_data_v99.groupby(["dow"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# ax = sns.scatterplot(x="dow", y="trip_dur", data=train_data_dow_v99)
# train_data_hour_v99 = train_data_v99.groupby(["hour"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# ax = sns.scatterplot(x="hour", y="trip_dur", data=train_data_hour_v99)
# bins = [-1, 3, 6, 9, 12, 15, 18, 21, 24]

# labels = [1,2,3,4,5,6,7,8]

# train_data_v99['hour_binned'] = pd.cut(train_data_v99['hour'], bins=bins, labels=labels)

# train_data_v99.head()
# train_data_hour_v99 = train_data_v99.groupby(["hour_binned"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# ax = sns.scatterplot(x="hour_binned", y="trip_dur", data=train_data_hour_v99)
# plt.figure(figsize=(15,8))

# train_data_v99["dow_hour_binned"] = train_data_v99.apply(lambda x : f"{x['dow']}_{x['hour_binned']}", axis=1)

# train_data_hour_dow_v99 = train_data_v99.groupby(["dow_hour_binned"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# ax = sns.scatterplot(x="dow_hour_binned", y="trip_dur", data=train_data_hour_dow_v99)
# train_data_hour_v99 = train_data_v99.groupby("hour").agg({'trip_duration': 'mean', 'id': 'count'})

# train_data_hour_v99.head()

# train_data_passen_v99 = train_data_v99.groupby(["passenger_count"])['trip_duration'].mean().to_frame(name = 'trip_dur').reset_index()

# ax = sns.scatterplot(x="passenger_count", y="trip_dur", data=train_data_passen_v99)
train_data = pd.read_csv('../input/featureadded/train_feat_add.csv')
def display_scores(scores):

    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))



def report_best_scores(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
train_data.head()
y = train_data['trip_duration']



features = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','distance','dow','hour']

X = train_data[features]

categorical_features = ['store_and_fwd_flag']

X = pd.get_dummies(X,columns =categorical_features)

X.pickup_longitude = X.pickup_longitude*(-1)

X.dropoff_longitude = X.dropoff_longitude*(-1)
X.head()
X_trim = X.reset_index(drop=True)

y_trim = y.reset_index(drop=True)



trip_dur_v99 = train_data['trip_duration'].quantile(0.99)

distance_v99 = train_data['distance'].quantile(0.99)



print(f"Originally X : {len(X_trim)}, Y : {len(y_trim)}")

X_trim = X_trim[list(y_trim)<=trip_dur_v99]

y_trim = y_trim[list(y_trim)<=trip_dur_v99]

print(f"After trip_duration X : {len(X_trim)}, Y : {len(y_trim)}")



X_trim = X_trim.reset_index(drop=True)

y_trim = y_trim.reset_index(drop=True)



y_trim = y_trim[(X_trim['distance']<=distance_v99)]

X_trim = X_trim[(X_trim['distance']<=distance_v99)]

print(f"After distance X : {len(X_trim)}, Y : {len(y_trim)}")



X_trim = X_trim.reset_index(drop=True)

y_trim = y_trim.reset_index(drop=True)



y_trim = y_trim[(X_trim['passenger_count']<=4)&(X_trim['passenger_count']>=1)]

X_trim = X_trim[(X_trim['passenger_count']<=4)&(X_trim['passenger_count']>=1)]

print(f"After passenger count X : {len(X_trim)}, Y : {len(y_trim)}")
xgb_model = xgb.XGBRegressor(objective="reg:squaredlogerror", random_state=42, verbosity=3)



xgb_model.fit(X_trim, y_trim, eval_metric='rmsle')



y_pred = xgb_model.predict(X_trim)



mse=mean_squared_log_error(y_trim, y_pred)



print(np.sqrt(mse))
xgb.plot_importance(xgb_model)
xgb_model = xgb.XGBRegressor(objective="reg:squaredlogerror", verbosity=2)



params = {

    "colsample_bytree": uniform(0.7, 0.3),

    "gamma": uniform(0, 0.5),

    "learning_rate": uniform(0.03, 0.3), # default 0.1 

    "max_depth": randint(2, 6), # default 3

    "n_estimators": randint(100, 150), # default 100

    "subsample": uniform(0.6, 0.4)

}



search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=20, cv=3, verbose=1, n_jobs=1, return_train_score=True)



search.fit(X_trim, y_trim, eval_metric='rmsle')



report_best_scores(search.cv_results_, 1)