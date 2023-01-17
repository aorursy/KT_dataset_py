import os 



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# Create a path to the data

path = os.path.join('..', 'input')

df_train = pd.read_csv(os.path.join(path, 'train.csv'))

df_train.shape
# Checking the data

df_train.head()
# Display the informations about the data

df_train.info()
# Checking some statistics informations about the data

df_train.describe()
ax, fig = plt.subplots(figsize=(12, 7))

sns.barplot(y='count', x='season', data=df_train, palette='hls')

plt.title('Repartion of the sharing by season');
ax, fig = plt.subplots(figsize=(12, 7))

sns.barplot(y='count', x='holiday', data=df_train, palette='hls')

plt.title('Repartion of the sharing by holiday');
ax, fig = plt.subplots(figsize=(12, 7))

sns.barplot(y='count', x='weather', data=df_train, palette='hls')

plt.title('Repartion of the sharing by weather types');
ax, fig = plt.subplots(figsize=(12, 7))

sns.barplot(y='count', x='workingday', data=df_train, palette='hls')

plt.title('Repartion of the sharing by working day');
# TConvert datetime column to a datetime

df_train['datetime'] = pd.to_datetime(df_train['datetime'])

df_train.head()
# Create a day of week column: dow

df_train['dow'] = df_train['datetime'].dt.dayofweek

df_train.head()
# Create a month column: month

df_train['month'] = df_train['datetime'].dt.month

df_train.head()
# Create a column with the week number: week

df_train['week'] = df_train['datetime'].dt.week

df_train.sample(10)
# Create a column with the hour: hour

df_train['hour'] = df_train['datetime'].dt.hour

df_train.head()
ax, fig = plt.subplots(figsize=(10, 7))

sns.barplot(y='count', x='month', data=df_train, palette='hls')

plt.title('Repartion of the sharing by month');
ax, fig = plt.subplots(figsize=(12, 7))

sns.barplot(y='count', x='dow', data=df_train, palette='hls')

plt.title('Repartion of the sharing by day of week');
ax, fig = plt.subplots(figsize=(12, 7))

sns.pointplot(y='count', x='hour', hue='dow', data=df_train, palette='hls')

plt.title('Repartion of the sharing by day of hour and day of the week');
ax, fig = plt.subplots(figsize=(15, 10))

sns.pointplot(y='count', x='dow', hue='month', data=df_train, palette='hls')

plt.title('Repartion of the sharing by day of month and day of the week');
# Define a y and X to work with by selecting the columns you're interested in. 

y_raw = df_train.loc[:, 'count']

X_raw = df_train.drop(['count', 'datetime', 'registered', 'casual'], axis=1)

y_raw.shape, X_raw.shape
# Import train_test_split to split your data.

from sklearn.model_selection import train_test_split
# Split the data and verify its shape.

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
# Display X_train

X_train.head()
# Verify the absence of null values

X_train.isna().sum()
# Display y_train

y_train[:5]
# Verify the absence of null values. 

y_train.isna().sum()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold
# Create a linear regression model and fit it on X_train, y_train.

lr = LinearRegression()

lr.fit(X_train, y_train)
# Predict on X_test.

y_pred = lr.predict(X_test)

y_pred
# Compute the mean squarred error.

mean_squared_error(y_test, y_pred)
X_train.max()
y_train.max()
# Import of the data.

df_test = pd.read_csv(os.path.join('..', 'input', 'test.csv'))

df_test.shape
# Checking the data.

df_test.head()
# Doing the same processing than on the train dataframe.

df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_test['month'] = df_test['datetime'].dt.month

df_test['week'] = df_test['datetime'].dt.week

df_test['dow'] = df_test['datetime'].dt.dayofweek

df_test['hour'] = df_test['datetime'].dt.hour

df_test_clear = df_test.drop(['datetime'], axis=1)

df_test_clear.sample(10)
X_test_2 = df_test.drop(['datetime'], axis=1)

X_test_2.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
# Create a RandomForestRegressor model

rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10, max_features='auto', 

max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, 

min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1, oob_score=False, 

random_state=None, verbose=0, warm_start=False)
# Fit the model on X_train and y_train

rf.fit(X_train, y_train)
# Predict on X_test and register the log of the prediction

log_pred = rf.predict(X_test_2)

y_pred = np.expm1(log_pred)

y_pred[:5]