import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot

from matplotlib.colors import LogNorm



%matplotlib inline

sns.set({'figure.figsize':(15,8)})



import os
# Load the training and testing datasets

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Print the first 5 rows of the traing dataset

df_train.head()
# Print the first 5 rows of the testing dataset

df_test.head()
# List all the training dataset features, the number of values in each and their type

df_train.info()
# List all the training dataset features, the number of values in each and their type

df_test.info()
# See if there are any missing values in the training dataset

for i, x in zip(list(df_train.isnull().sum().index),list(df_train.isnull().sum().values)):

    print(f"The feature {i} counts {x} missing value")
# See if there are any missing value in the testing dataset as well

for i, x in zip(list(df_test.isnull().sum().index),list(df_test.isnull().sum().values)):

    print(f"The feature {i} counts {x} missing value")
# Now let's make sure there are no duplicated values

train_mv = df_train.duplicated().sum()

test_mv = df_test.duplicated().sum()

print(f'The training dataset counts {train_mv} duplicated value and the testing dataset counts {test_mv} as well.')
# Split the cells content in two new features: pickup_day and pickup_time

df_train['pickup_day'] = pd.to_datetime(df_train['pickup_datetime']).dt.date

df_train['pickup_time'] = pd.to_datetime(df_train['pickup_datetime']).dt.time



# We apply this same logic to the testing set

df_test['pickup_day'] = pd.to_datetime(df_test['pickup_datetime']).dt.date

df_test['pickup_time'] = pd.to_datetime(df_test['pickup_datetime']).dt.time



# Then we do the same for the dropoff feature

df_train['dropoff_day'] = pd.to_datetime(df_train['dropoff_datetime']).dt.date

df_train['dropoff_time'] = pd.to_datetime(df_train['dropoff_datetime']).dt.time
df_train.head()
df_train['trip_duration'].describe()
# Now that we now the mean and median, let's see if there are outliers

df_train.boxplot(['trip_duration']);
df_train.boxplot(['trip_duration'], showfliers=False, notch=True);
len(df_train.trip_duration[df_train.trip_duration > 4000].values)
len(df_train.trip_duration[df_train.trip_duration < 10].values)
# Let's create a new dataframe, without the outliers

df2_train = df_train[df_train.trip_duration < 4000]

df2_train.info()
# Let's see the values changes 

df2_train['trip_duration'].describe()
df2_train['trip_duration'].hist(bins=100, histtype='stepfilled')

plt.title("Ditribution of the trip_duration feature points");
df2_train.boxplot(['pickup_longitude', 'dropoff_longitude', 'pickup_latitude', 'dropoff_latitude']);
len(df2_train.pickup_longitude[df2_train.pickup_longitude < -80].values)
len(df2_train.pickup_longitude[df2_train.pickup_longitude > -50].values)
len(df2_train.pickup_latitude[df2_train.pickup_latitude < 25].values)
len(df2_train.pickup_latitude[df2_train.pickup_latitude > 50].values)
min_long = -80

max_long = -50

min_lat = 25

max_lat = 50
fig = plt.figure(1, figsize=(10,5))

hist = plt.hist2d(df2_train.pickup_longitude, df2_train.pickup_latitude, bins=50, range=[[min_long,max_long], [min_lat,max_lat]], norm=LogNorm())

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.colorbar(label='Value counts')

plt.title('NYC pickup locations')

plt.show()
num_col = [

    col for col in df2_train.columns if 

    (df2_train[col].dtype=='int64' or df2_train[col].dtype=='float64') 

    and col != 'trip_duration']



num_col
cat_col = [

    col for col in df2_train.columns if 

    (df2_train[col].dtype=='object') 

    and col != 'trip_duration']



cat_col
for col in cat_col:

    df2_train[col] = df2_train[col].astype('category').cat.codes

    

df2_train.head()
# Finally, we lock the target fature in a constant one

TARGET = df2_train.trip_duration
df2_train.head()
X_train = df2_train[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_day', 'pickup_time']]

X_train.shape
y_train = df2_train.trip_duration

y_train.shape
from sklearn.model_selection import train_test_split
X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, 

                                                              test_size=.2, random_state=42, stratify=y_train)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
from sklearn.ensemble import RandomForestRegressor
m1 = RandomForestRegressor(n_estimators=10, random_state=6)

m1.fit(X_train, y_train)

m1.score(X_valid, y_valid)
m2 = RandomForestRegressor(n_estimators=5, random_state=42)

m2.fit(X_train, y_train)

m2.score(X_valid, y_valid)
m3 = RandomForestRegressor(n_estimators=8, random_state=42, max_leaf_nodes=100)

m3.fit(X_train, y_train)

m3.score(X_valid, y_valid)
m4 = RandomForestRegressor(n_estimators=10, random_state=42, max_leaf_nodes=750)

m4.fit(X_train, y_train)

m4.score(X_valid, y_valid)
m5 = RandomForestRegressor(n_estimators=10, random_state=42, max_leaf_nodes=50000)

m5.fit(X_train, y_train)

m5.score(X_valid, y_valid)
from sklearn.metrics import r2_score
y_valid_pred = m5.predict(X_valid)

y_valid_pred
r2_score(y_valid, y_valid_pred)
from sklearn.metrics import mean_squared_error as MSE

from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train, y_train)
MSE(y_valid, m5.predict(X_valid))
loss = MSE(y_valid, sgd.predict(X_valid))

loss
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)

MSE(y_valid, lr.predict(X_valid))
from sklearn.model_selection import cross_val_score



cv_scores = cross_val_score(m5, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

cv_scores
X_test = df_test[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_day', 'pickup_time']]

y_test_pred = m5.predict(X_test)

y_test_pred[:10]
submission = pd.DataFrame(df_test.loc[:, 'id'])

submission['trip_duration'] = y_test_pred

print(submission.shape)

submission.head()
submission.to_csv("submit_file.csv", index=False)