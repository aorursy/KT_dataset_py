import re
import sys
import os

import time
import datetime

import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
apps = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
apps.head()
apps.describe()
apps.shape
#Setting options to display all rows and columns

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.width=None
# missing data
total = apps.isnull().sum().sort_values(ascending=False)
percent = (apps.isnull().sum()/apps.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(4)
# The best way to fill missing values might be using the median instead of mean.
apps['Rating'] = apps['Rating'].fillna(apps['Rating'].median())
# Before filling null values we have to clean all non numerical values & unicode charachters 
replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
for i in replaces:
    apps['Current Ver'] = apps['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
for j in regex:
    apps['Current Ver'] = apps['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

apps['Current Ver'] = apps['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)
apps['Current Ver'] = apps['Current Ver'].fillna(apps['Current Ver'].median())
# Unwanted record of caetgory - which is 1.9
apps.Category.unique()
# Check the record  of unreasonable value which is 1.9
i = apps[apps['Category'] == '1.9'].index
apps.loc[i]
# Drop this bad column
apps = apps.drop(i)
# Removing NaN values
apps = apps[pd.notnull(apps['Last Updated'])]
apps = apps[pd.notnull(apps['Content Rating'])]
# Contribution in null value by Cuurent Ver, Android ver, Content Rating & Type is hardly make .5% so it's better to drop null values of them. 
apps.dropna(how='any', inplace=True)
# Changed dimension:
apps.shape
apps = apps[apps['Content Rating'] != 'Unrated']
apps['Content Rating'].unique()
# Content rating features encoding
content_list = apps['Content Rating'].unique().tolist() 
content_list = ['con_' + word for word in content_list]
apps = pd.concat([apps, pd.get_dummies(apps['Content Rating'], prefix='con')], axis=1)
apps['Price'] = apps['Price'].str.replace('$',"")
apps['Price'] = apps['Price'].apply(lambda x: float(x))
# App values encoding
le = preprocessing.LabelEncoder()
apps['App'] = le.fit_transform(apps['App'])
# This encoder converts the values into numeric values
apps.head()
# Category features encoding
category_list = apps['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
apps = pd.concat([apps, pd.get_dummies(apps['Category'], prefix='cat')], axis=1)

apps['Genres'] = apps['Genres'].str.split(';').str[0]
apps['Genres'].value_counts().tail()
apps['Genres'].replace('Music & Audio', 'Music',inplace = True)
apps['Genres'].value_counts().tail()
# Genres features encoding
le = preprocessing.LabelEncoder()
apps['Genres'] = le.fit_transform(apps['Genres'])
apps.head(2)
# Removing punchuations ',' & '+' sign from Intalls column:
apps.Installs = apps.Installs.apply(lambda x: x.replace(',',''))
apps.Installs = apps.Installs.apply(lambda x: x.replace('+',''))
apps.Installs = apps.Installs.apply(lambda x: int(x))
apps.head(2)
# Type encoding
apps['Type'] = pd.get_dummies(apps['Type'])
# Convert kbytes to Mbytes 
k_indices = apps['Size'].loc[apps['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(apps.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
apps.loc[k_indices,'Size'] = converter
# Size cleaning
apps['Size'] = apps['Size'].apply(lambda x: x.strip('M'))
apps[apps['Size'] == 'Varies with device'] = 0
apps['Size'] = apps['Size'].astype(float)
apps.head()
# I'm just not considering this parameter for my model building
apps['Reviews'] = apps['Reviews'].apply(lambda x: float(x))
# Split data into training and testing sets
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Genres', 'Current Ver']
features.extend(category_list)
features.extend(content_list)
X = apps[features]
y = apps['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)
# Look at the 10 closest neighbors
model = KNeighborsRegressor(n_neighbors=10)
# Find the mean accuracy of knn regression using X_test and y_test
model.fit(X_train, y_train)
# Calculate the mean accuracy of the KNN model
accuracy = model.score(X_test,y_test)
'Accuracy of this model is ' + str(np.round(accuracy*100, 2)) + '%'
# We can try with different numbers of n_estimators or combination of various neighbors - this will take a minute or so
n_neighbors = np.arange(1, 22, 1)
scores = []
for n in n_neighbors:
    model.set_params(n_neighbors=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(8, 6))
plt.title("Effect of Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neighbors, scores)
model = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results

predictions = model.predict(X_test)
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)
'Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)
'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))
from sklearn.metrics import accuracy_score
model3 = XGBRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 50, 2)
scores = []
for n in estimators:
    model3.set_params(n_estimators=n)
    model3.fit(X_train, y_train)
    scores.append(model3.score(X_test, y_test))
plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results