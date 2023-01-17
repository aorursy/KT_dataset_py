# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

visualization = train_data.copy()
visualization.datetime = pd.to_datetime(visualization.datetime)
visualization.head()
# Visualization

date = visualization[['datetime', 'count']]
date.is_copy = False
date = date.groupby(pd.TimeGrouper(key='datetime',freq='M'))
date.sum().plot(figsize=(12,6))
seasons = visualization[['datetime','season','count']].rename(columns={'datetime':'year'})
seasons.is_copy = False
seasons.year = seasons.year.dt.year
fig,(ax) = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(11,10))
seasons[seasons['year']==2012].groupby(seasons['season']).mean().plot(kind='bar',x='season',y='count',ax=ax,legend=False)
ax.set_title('Demand over seasons')
ax.set_xticklabels(("Winter", "Spring", "Summer", "Fall"))
plt.tight_layout()
weekday = visualization[['datetime', 'count']]
weekday.is_copy = False
weekday['weekday'] = weekday.datetime.dt.dayofweek
fig,(ax) = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(11,7))
weekday.groupby(weekday.weekday).mean().plot(x='weekday',y='count',ax=ax,label='Total')
ax.set_xticklabels(("Monday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
plt.tight_layout()
hours = visualization[['datetime', 'count']]
hours.is_copy = False
hours['hour'] = hours.datetime.dt.hour
fig,(ax) = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(11,7))
hours.groupby(hours.hour).mean().plot(x='hour',y='count',ax=ax,label='Total')
weather = visualization[['weather','count']]
weather.is_copy = False
fig,(ax) = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(11,7))
weather.groupby(weather.weather).mean().plot(stacked=True,kind='bar', figsize=(11,5),legend=False, ax=ax)
ax.set_xticklabels(("Clear", "Cloudy", "Light rain", "Heavy rain"))
train_data.datetime = train_data.datetime.apply(pd.to_datetime)
train_data['year'] = train_data.datetime.apply(lambda x: x.year)
train_data['month'] = train_data.datetime.apply(lambda x: x.month)
train_data['day'] = train_data.datetime.apply(lambda x: x.day)
train_data['hour'] = train_data.datetime.apply(lambda x: x.hour)
train_data.drop('datetime', axis=1, inplace=True)
train_data.head()
test_data.datetime = test_data.datetime.apply(pd.to_datetime)
test_data['year'] = test_data.datetime.apply(lambda x: x.year)
test_data['month'] = test_data.datetime.apply(lambda x: x.month)
test_data['day'] = test_data.datetime.apply(lambda x: x.day)
test_data['hour'] = test_data.datetime.apply(lambda x: x.hour)
test_data.drop('datetime', axis=1, inplace=True)
test_data.head()
train_data.head()
# Grouping categorical features
categorical_columns = ['season','holiday','workingday','weather', "month", "day", "hour"]
df_categorical_columns = train_data[categorical_columns]
categorical_feat_dict = df_categorical_columns.T.to_dict().values()


# Grouping non categorical features
noncategorical_columns = ['temp','humidity','windspeed']
df_noncategorical_columns = train_data[noncategorical_columns]
noncategorical_feat_dict = df_noncategorical_columns.T.to_dict().values()
# Vectorizing feature groups
vectorizer = DictVectorizer(sparse = False)
categorical_vector = vectorizer.fit_transform(categorical_feat_dict)

vectorizer = DictVectorizer(sparse = False)
noncategorical_vector = vectorizer.fit_transform(noncategorical_feat_dict)

# Encoding feature vectors
encoder = OneHotEncoder()
encoder.fit(categorical_vector)
categorical_vector = encoder.transform(categorical_vector).toarray()
# Combining noncategorical and categorical data
x = np.concatenate((categorical_vector, noncategorical_vector), axis=1)
x
# Splitting test and training data
y = train_data["count"]
y_registered = train_data["registered"]
y_casual = train_data["casual"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
len(x_train)
len(x_test)
# RMSLE for performance measuring
def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(p + 1) for p in y]))
    log2 = np.nan_to_num(np.array([np.log(a + 1) for a in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
start_time = time.time()
regressor = RandomForestRegressor()
param_grid = {'n_estimators': np.arange(40, 50)}
grid_random_forest = GridSearchCV(regressor, param_grid, cv=5)
grid_random_forest.fit(X=x_train, y= np.log1p(y_train))
y_pred = grid_random_forest.predict(X=x_test)
elapsed_time = time.time() - start_time
print("Random Forrest")
print("Time", elapsed_time, "seconds")
print("RMSLE: ", rmsle(y_test,np.exp(y_pred)))
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(train_data)
pca_dataset = pca.transform(train_data)
from sklearn import linear_model

x_train, x_test, y_train, y_test = train_test_split(pca_dataset, y, test_size = 0.2, random_state=42)
linear_regressor = linear_model.LinearRegression().fit(x_train, np.log1p(y_train))
start_time = time.time()
prediction = linear_regressor.predict(x_test)
elapsed_time = time.time() - start_time
print("Linear Regression")
print("Time", elapsed_time, "seconds")
print("RMSLE: ", rmsle(y_test, np.exp(prediction)))