# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from __future__ import unicode_literals
import pandas as pd
%matplotlib inline
import re
import sys
import time
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
"""
Data Loading
"""
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv",encoding='utf-8')
data.head(5)
"""
Data Cleaning & Exploratory Data Analysis (EDA)
"""
data.isnull().sum()
data.isnull().any()
plt.figure(figsize=(7, 5))
sns.heatmap(data.isnull())
data.isnull().any()
data.describe()
data['Rating'] = data['Rating'].fillna(data['Rating'].median())
data['Current Ver'] = data['Current Ver'].replace('Varies with device',np.nan)
data['Current Ver'] = data['Current Ver'].fillna(data['Current Ver'].mode()[0])

# Count number of unique values in category column 
data['Category'].unique()

# Removing NaN values
data = data[pd.notnull(data['Last Updated'])]
data = data[pd.notnull(data['Content Rating'])]
# This is to be anomaly record.
i = data[data['Category'] == '1.9'].index
data.loc[i]
# Drop the anomaly record
data = data.drop(i)
data
"""
Feature Engineering using Label Encoder and converting categorical data to dummy variables.
"""
# App values encoding

le = preprocessing.LabelEncoder()
data['App'] = le.fit_transform(data['App'])

# Category features encoding

category_list = data['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
category_list
data = pd.concat([data, pd.get_dummies(data['Category'], prefix='cat')], axis=1)

# Genres features encoding
le = preprocessing.LabelEncoder()
data['Genres'] = le.fit_transform(data['Genres'])

# Encode Content Rating features
le = preprocessing.LabelEncoder()
data['Content Rating'] = le.fit_transform(data['Content Rating'])

# Price cleaning
data['Price'] = data['Price'].apply(lambda x : x.strip('$'))
# Installs cleaning
data['Installs'] = data['Installs'].apply(lambda x : x.strip('+').replace(',', ''))

# Type encoding
data['Type'] = pd.get_dummies(data['Type'])

# Last Updated encoding
data['Last Updated'] = data['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))

# Convert kbytes to mbytes 
k_indices = data['Size'].loc[data['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(data.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
data.loc[k_indices,'Size'] = converter

# Size cleaning
data['Size'] = data['Size'].apply(lambda x: x.strip('M'))
data[data['Size'] == 'Varies with device'] = 0
data['Size'] = data['Size'].astype(float)

"""
Split Data into Training and Testing DataSet & Model Building.
"""


# Split data into training and testing sets
features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated']
features.extend(category_list)

X = data[features]
Y = data['Rating']

# Split the dataset into 75% train data and 25% test data.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 10)

model = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 450, 10)
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
"""
Predictions & Model Evaluation
"""
predictions = model.predict(X_test)
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)

'Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)

'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))

"""
From the above grid search 
"""
model = RandomForestRegressor(n_estimators=20, 
                               bootstrap = True)
# Fit on training data
model.fit(X_train, y_train)

rf_predictions = model.predict(X_test)
rf_predictions
