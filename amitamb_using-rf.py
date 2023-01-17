# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
n_train = 1_000_000

n_features = 2

dtype_map = {

    'fare_amount': np.float32,

    'pickup_longitude': np.float64,

    'pickup_latitude': np.float64,

    'dropoff_longitude': np.float64,

    'dropoff_latitude': np.float64,

    'passenger_count': np.int8

}



df = pd.read_csv("../input/train.csv", nrows = n_train, parse_dates=["pickup_datetime"], dtype=dtype_map)



df_test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"], dtype=dtype_map)
df.describe()
df.dtypes
def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

    df['circ_diff'] = np.sqrt(df.abs_diff_longitude**2 + df.abs_diff_latitude**2)



add_travel_vector_features(df)

add_travel_vector_features(df_test)

print(df.isnull().sum())

print('Old size: %d' % len(df))

df = df.dropna(how='any', axis='rows')

print('New size: %d' % len(df))
df.head()
# print(train_df.pickup_datetime.min())

# print(train_df.pickup_datetime.apply(lambda t: t.year).min())

min_year = df.pickup_datetime.apply(lambda t: t.year).min()

df['pickup_year'] = df.pickup_datetime.apply(lambda t: t.year - min_year)

df['pickup_hour'] = df.pickup_datetime.apply(lambda t: t.hour)

df['pickup_day'] = df.pickup_datetime.apply(lambda t: t.timetuple().tm_yday)



df_test['pickup_year'] = df_test.pickup_datetime.apply(lambda t: t.year - min_year)

df_test['pickup_hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)

df_test['pickup_day'] = df_test.pickup_datetime.apply(lambda t: t.timetuple().tm_yday)

# print(train_df.iloc[0].pickup_datetime.timetuple())

# train_df.dtypes
from sklearn.model_selection import train_test_split



df_train, df_val = train_test_split(df, test_size=0.1)

len(df_val)
df_train.head()
def get_input_matrix(df):

    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, df.circ_diff,

                            df.dropoff_longitude, df.dropoff_latitude,

                            df.pickup_longitude, df.pickup_latitude,

                            df.passenger_count, df.pickup_year,

                            df.pickup_hour, df.pickup_day))



x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)

y_train, y_val = np.array(df_train.fare_amount), np.array(df_val.fare_amount)
from sklearn.ensemble import RandomForestRegressor

# ??RandomForestRegressor

reg = RandomForestRegressor(max_depth=28, n_estimators=500, oob_score=True, n_jobs=-1,min_samples_split=10, verbose=1)
reg.fit(x_train, y_train)
reg.oob_score_
from sklearn.metrics import r2_score

y_pred = reg.predict(x_val)

# score = accuracy_score(y_val, y_pred)

# ??accuracy_score
score = r2_score(y_val, y_pred)

score
%matplotlib inline

import matplotlib.pyplot as plt



plt.scatter(y_val, y_pred, alpha=0.02)
reg.score(x_val, y_val)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred)
#Read and preprocess test set

# df_test =  pd.read_csv('../input/test.csv')

# df_test = add_datetime_info(test_df)

# df_test = add_airport_dist(test_df)

# df_test['distance'] = sphere_dist(df_test['pickup_latitude'], df_test['pickup_longitude'], 

#                                    df_test['dropoff_latitude'] , df_test['dropoff_longitude'])



# test_key = df_test['key']

# x_pred = df_test.drop(columns=['key', 'pickup_datetime'])



# #Predict from test set

# prediction = reg.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
x_test = get_input_matrix(df_test)
predictions = reg.predict(x_test)
RFSubmission = pd.DataFrame({ 'key': df_test.key.ravel(),

                            'fare_amount': predictions })

RFSubmission.to_csv("RFSubmission.csv", index=False)