# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/nyc-taxi-trip-duration"))



# Any results you write to the current directory are saved as output.
# !ls ../input/nytaxi-clean-pairplot
!ls ../input/nytaxi-clean-pairplot/pairplot.png
# Import libraries

import datetime

import math



import geopy.distance



import seaborn as sns

import matplotlib.pyplot as plt





%matplotlib inline
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',

                   parse_dates=['pickup_datetime', 'dropoff_datetime'],

                   dtype={'store_and_fwd_flag':'category'})
train.info()
train.head()
train.shape
def create_datetime_columns(df, column_list):

    

    for col_name in column_list:

        df[col_name+ '_' + 'dayofweek'] = df[col_name].dt.dayofweek

        df[col_name+ '_' + 'dayofyear'] = df[col_name].dt.dayofyear

        df[col_name+ '_' + 'weekofyear'] = df[col_name].dt.weekofyear

        df[col_name+ '_' + 'month'] = df[col_name].dt.month

        df[col_name+ '_' + 'hour'] = df[col_name].dt.hour

        

    return df
def get_distance_km(row):

    coords_1 = (row.pickup_latitude, row.pickup_longitude)

    coords_2 = (row.dropoff_latitude, row.dropoff_longitude)

    

    return geopy.distance.geodesic(coords_1, coords_2).km
def transform_data(df, cleanData=False):

    

    data_clean = df.copy()

    

    #### Categorical column (store_and_fwd_flag)

    # This column must be converted to a numerical value by 

    # using cat.codes and cast it to int

    data_clean['store_and_fwd_flag'] = data_clean['store_and_fwd_flag'].cat.codes



    #### Datetime columns (pickup_datetime)

    # datetime columns which is **pickup_datetime**

    # should be split to 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'hour'

#     data_clean = create_datetime_columns(data_clean, 

#                                          ['pickup_datetime', 'dropoff_datetime'])

    # Only do get additional column for pickup_datetime should be enought because

    # They are typically on the same day

    data_clean = create_datetime_columns(data_clean, 

                                         ['pickup_datetime'])



    #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)

    # Create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)

    data_clean['distance_km'] = data_clean.apply(lambda row: get_distance_km(row), axis=1)

       

    if cleanData:

        # After doing the exploratory analysis, I found that there are outliers in the dataset

        # (there are trips that have 1k km) that could potentially cause an unexpected behavior

        # Hence, remove those outlier data before proceeding         

        data_clean = data_clean[data_clean.distance_km < data_clean.distance_km.quantile(0.99)]

    

    

    #### Drop unused columns

    # **id** column can be dropped because we do not need it in training

    # **pickup_datetime** and **dropoff_datetime** must be dropped after all above are done



    data_clean = data_clean.drop(['id', 

                                  'pickup_datetime'

                                 ], axis=1)

    

    # Test data does not have dropof_datetime column. Hence, skip it

    if data_clean.columns.contains('dropoff_datetime'):

        data_clean = data_clean.drop(['dropoff_datetime'], axis=1)

    

    return data_clean
%time data_clean = transform_data(train, cleanData=True)

data_clean.reset_index().to_feather('data_clean')
# # # We will use a saved clean data from the previous session here

# data_clean = pd.read_feather('../input/nytaxi-clean-feather/data_clean')
data_clean.head()
# Inspect the output dataframe

data_clean.sample(20)
data_clean.info()
corr = data_clean.corr()
plt.figure(figsize=(8,6))



sns.heatmap(corr);
corr.style.background_gradient(cmap='coolwarm')
# # sns_plot = sns.pairplot(df)

# # sns_plot.show()



# # Load a picture of the pairplot generated from the command above instead since

# # it takes a long time (around two hours) to create it

# from IPython.display import Image

# Image("../input/nytaxi-clean-pairplot/pairplot.png")
# Get all column names

data_clean.columns.tolist()
data_clean['vendor_id'].value_counts()
data_clean['passenger_count'].hist();
data_clean['passenger_count'].plot.box();
data_clean['passenger_count'].describe()
sns.countplot(x="passenger_count", hue="vendor_id", data=data_clean);
data_clean.distance_km.describe()
data_clean.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
X = data_clean.drop(['trip_duration'], axis=1)

y = data_clean['trip_duration']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=42)
X_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m, X_train, X_valid, y_train, y_valid):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)



print('[{}] Start'.format(datetime.datetime.now()))



%time m.fit(X_train, y_train)

print('[{}] Start'.format(datetime.datetime.now()))





%time print_score(m, X_train, X_valid, y_train, y_valid)
y_pred_train = m.predict(X_train)
y_pred = m.predict(X_valid)
# From https://stackoverflow.com/questions/46202223/root-mean-log-squared-error-issue-with-scitkit-learn-ensemble-gradientboostingre

def rmsle(y, y0):

    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
rmsle(y_train, y_pred_train)
rmsle(y_valid, y_pred)
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',

                   parse_dates=['pickup_datetime'],

                   dtype={'store_and_fwd_flag':'category'})
test.head()
test.shape
%time test_clean = transform_data(test)
test_clean.head()
test_clean.info()
test_clean.shape
X_sub = test_clean.copy()
y_sub = m.predict(X_sub)
df_sub = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')
df_sub.head()
df_sub['trip_duration'] = y_sub

df_sub.head()
df_sub.shape
df_sub.to_csv('submission_default_randomforest.csv', index=False)