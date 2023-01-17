# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Lasso
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read the training data
# train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)

train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0, 16)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
train_df.head()
train_df.loc[:,('dropoff_longitude','pickup_longitude')].head()
print(train_df.dropoff_longitude[:2])
print(train_df.isnull().sum())
train_df = train_df[train_df.fare_amount>=0]

#Remove NaN from dataset
print('Old size %d'% len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))
#change add_traval_vector_features to calculate distance using  Haversine 
# def add_travel_vector_features(df):
#     df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
#     df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

# add_travel_vector_features(train_df)
def distance(df):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - np.cos((df.dropoff_latitude - df.pickup_latitude) * p)/2 + np.cos(df.pickup_latitude * p) * np.cos(df.dropoff_latitude * p) * (1 - np.cos((df.dropoff_longitude - df.pickup_longitude) * p)) / 2
#     return 12742 * asin(sqrt(a)) #2*R*asin...
    df['distance_miles'] = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
distance(train_df)
# plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
# print('Old size: %d' % len(train_df))
# train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
# print('New size: %d' % len(train_df))
train_df = train_df[(train_df.distance_miles < 345)]

# Removing observations with erroneous values
mask = train_df['pickup_longitude'].between(-75, -73)
mask &= train_df['dropoff_longitude'].between(-75, -73)
mask &= train_df['pickup_latitude'].between(40, 42)
mask &= train_df['dropoff_latitude'].between(40, 42)
mask &= train_df['passenger_count'].between(0, 8)
mask &= train_df['fare_amount'].between(0, 250)

train_df = train_df[mask]


train_df['hour'] = train_df.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
train_df['year'] = train_df.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
# Get input matrix
def get_input_matrix(df):
    return np.column_stack((df.year,df.hour,df.distance_miles,df.passenger_count, np.ones(len(df))))


#X_train= get_input_matrix(train_df)
#y_train = np.array(train_df['fare_amount'])

#print(X_train.shape)
#print(y_train.shape)
X = get_input_matrix(train_df)
y = np.array(train_df['fare_amount'])

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# Create the regressor: reg_all
reg_all = LinearRegression()
cv_scores = cross_val_score(reg_all,X,y,cv=5)
# Fit the regressor to the training data
reg_all.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


train_df.columns.values
df_columns = pd.Index([  'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'distance_miles','hour','year'])
df_columns
#Find the most important variable
# Instantiate a lasso regressor: lasso
X_all_feature = np.column_stack((train_df.pickup_longitude,train_df.pickup_latitude,train_df.dropoff_longitude,
                    train_df.dropoff_latitude,train_df.passenger_count,train_df.distance_miles,train_df.hour,train_df.year))
X_all_feature
lasso = Lasso(alpha=0.1)

# Fit the regressor to the data
lasso.fit(X_all_feature,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
test_df = pd.read_csv('../input/test.csv')
test_df.dtypes
# Reuse the above helper functions to add our features and generate the input matrix.
# add_travel_vector_features(test_df)
distance(test_df)
test_df['hour'] = test_df.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
test_df['year'] = test_df.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
test_X = get_input_matrix(test_df)

# Predict fare_amount on the test set using our model (w) trained on the training set.
# test_y_predictions = np.matmul(test_X, w_OLS).round(decimals = 2)
test_y_predictions = reg_all.predict(test_X)
test_y_predictions

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
test_y_predictions[:200]
data = pd.read_csv('submission.csv')
data