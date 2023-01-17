# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data viz.
import seaborn as sb #data viz.
from sklearn.ensemble import GradientBoostingRegressor #ML algorithm
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train =  pd.read_csv('../input/train.csv', nrows = 100_000, parse_dates=["pickup_datetime"])

# list first few rows (datapoints)
df_train.head()

df_my_test = df_train.copy()
df_my_test['distance'] = np.square(df_my_test['pickup_longitude'] - df_my_test['dropoff_longitude']) + np.square(df_my_test['pickup_latitude'] - df_my_test['dropoff_latitude'])
df_my_test.head()
nyc = (-74.0063889, 40.7141667)
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
df_my_test['distance_miles'] = distance(df_my_test.pickup_latitude, df_my_test.pickup_longitude, \
                                     df_my_test.dropoff_latitude, df_my_test.dropoff_longitude)
df_my_test['distance_to_center'] = distance(nyc[1], nyc[0], \
                                          df_my_test.dropoff_latitude, df_my_test.dropoff_longitude)
#df_my_test['hour'] = df_my_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
#df_my_test['year'] = df_my_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
df_my_test['distance_miles'] > 1000 
X_train = df_my_test[['distance_miles', 'distance_to_center','passenger_count']]
uh = df_my_test['hour'] < 8
X_train['costly'] = uh
X_train.head()
uh.head()
idxs = df_my_test['distance'] > 0.1
df_my_test.loc[idxs,'distance'] = 0
idxs = df_my_test['distance'] > 0.1
df_my_test[idxs]
df_my_test['norm_distance'] = df_my_test['distance'] / max(df_my_test['distance'])
df_my_test.head()
max(df_my_test['distance'])
X_train = df_my_test[['distance_miles','passenger_count']]
X_train.head()

import datetime as dt

df_my_test['Year'] = df_my_test['pickup_datetime'].dt.year
df_my_test['Month'] = df_my_test['pickup_datetime'].dt.month
df_my_test['Date'] = df_my_test['pickup_datetime'].dt.day
df_my_test['Day_of_Week'] = df_my_test['pickup_datetime'].dt.dayofweek
df_my_test['Hour'] = df_my_test['pickup_datetime'].dt.hour
df_my_test.head()
X_train['time'] = df_my_test['']
X_train.head()
y_train = df_my_test['fare_amount']
y_train.head()
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
   #     ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)

#y_test_pred = model_lin.predict(X_test)

from sklearn.metrics import mean_squared_error
    
rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
rmse
y_train[1:5]
y_train_pred[1:5]
# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df_my_test)
df_my_test.head()
#analyzing the distribution of `Fair amount`
plt.hist(df_my_test['fare_amount'][:2000])

plt.xlabel('Fair amount in dollars')
plt.show()
df_my_test.columns
#Write a function to get the columns that we want to include in our X matrix as we would be doing the same with our test set.
def get_input_matrix(df):
    return np.column_stack((df.pickup_longitude, df.abs_diff_longitude, df.abs_diff_latitude, 
                            df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude, 
                            df.Hour, df.Day_of_Week, df.Month, df.Year))

X = get_input_matrix(df_my_test)
Y = np.array(df_my_test['fare_amount'])

print(X.shape)
print(Y.shape)
# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df_my_test)
df_my_test.head()
plot = df_my_test.iloc[:10000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print('Old size: %d' % len(df_my_test))
df_my_test = df_my_test[(df_my_test.abs_diff_longitude < 5.0) & (df_my_test.abs_diff_latitude < 5.0)]
print('New size: %d' % len(df_my_test))
#analyzing the distribution of `Fair amount`
plt.hist(df_my_test['fare_amount'][:2000])

plt.xlabel('Fair amount in dollars')
plt.show()
#Write a function to get the columns that we want to include in our X matrix as we would be doing the same with our test set.
def get_input_matrix(df):
    return np.column_stack((df.pickup_longitude, df.abs_diff_longitude, df.abs_diff_latitude, 
                            df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude, 
                            df.Hour, df.Day_of_Week, df.Month, df.Year))

X = get_input_matrix(df_my_test)
Y = np.array(df_my_test['fare_amount'])

print(X.shape)
print(Y.shape)
#Divide our data into train and validation set. We will be using validation set to tune the hyperparameters of the model. 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.0005, random_state=0)

print(X_train.shape)
print(X_validation.shape)
print(Y_train.shape)
print(Y_validation.shape)
# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [1]
for learning_rate in learning_rates:
    gb = GradientBoostingRegressor(n_estimators = 10, learning_rate = learning_rate, max_depth = 6, random_state = 0)
    gb.fit(X_train, Y_train)
    pred_train = gb.predict(X_train)
    pred_validation = gb.predict(X_validation)
    print("Learning rate: ", learning_rate)
    print("RMSE (training): {0:.3f}".format(np.sqrt(mean_squared_error(Y_train, pred_train))))
    print("RMSE (validation): {0:.3f}".format(np.sqrt(mean_squared_error(Y_validation, pred_validation))))
    print()
#let's see what are the significant features in predicting our output
gb.feature_importances_
plt.bar(range(len(gb.feature_importances_)), gb.feature_importances_)
plt.show()
df_my_test.head()
test_X = get_input_matrix(df_my_test)
test_y_predictions = gb.predict(test_X)

ss = pd.read_csv('../input/sample_submission.csv')
ss.head()
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

df_train_wo_dates = df_my_test.drop(['key', 'pickup_datetime'], axis=1)

sourcevars = df_train_wo_dates.values[:,:-1]
targetvar = df_train_wo_dates.values[:,len(df_train_wo_dates.values[0])-1]

X_train, X_test, y_train, y_test = train_test_split(sourcevars, targetvar, test_size=0.2)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)

scores = cross_val_score(model, sourcevars, targetvar, cv=6)
'Cross-validated scores:', scores

predictions = cross_val_predict(model, sourcevars, targetvar, cv=6)
plt.scatter(targetvar, predictions)

accuracy = metrics.r2_score(targetvar, predictions)
accuracy
