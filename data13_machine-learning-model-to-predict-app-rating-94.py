import re

import sys



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

from sklearn.model_selection import train_test_split



# Loading the data

df = pd.read_csv('../input/googleplaystore.csv')
%matplotlib inline
sns.set(style='darkgrid')

sns.set_palette('PuBuGn_d')
df.head()

# Executing the above script will display the first five rows of the dataset as shown below
# Checking the data type of the columns

df.info()
# Exploring missing data and checking if any has NaN values

plt.figure(figsize=(7, 5))

sns.heatmap(df.isnull(), cmap='viridis')

df.isnull().any()
df.isnull().sum()
# The best way to fill missing values might be using the median instead of mean.

df['Rating'] = df['Rating'].fillna(df['Rating'].median())



# Before filling null values we have to clean all non numerical values & unicode charachters 

replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]

for i in replaces:

	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))



regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']

for j in regex:

	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))



df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)

df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())
# Count the number of unique values in category column 

df['Category'].unique()
# Check the record  of unreasonable value which is 1.9

i = df[df['Category'] == '1.9'].index

df.loc[i]
# Drop this bad column

df = df.drop(i)
# Removing NaN values

df = df[pd.notnull(df['Last Updated'])]

df = df[pd.notnull(df['Content Rating'])]
# App values encoding

le = preprocessing.LabelEncoder()

df['App'] = le.fit_transform(df['App'])

# This encoder converts the values into numeric values
# Category features encoding

category_list = df['Category'].unique().tolist() 

category_list = ['cat_' + word for word in category_list]

df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)
# Genres features encoding

le = preprocessing.LabelEncoder()

df['Genres'] = le.fit_transform(df['Genres'])
# Encode Content Rating features

le = preprocessing.LabelEncoder()

df['Content Rating'] = le.fit_transform(df['Content Rating'])
# Price cealning

df['Price'] = df['Price'].apply(lambda x : x.strip('$'))
# Installs cealning

df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
# Type encoding

df['Type'] = pd.get_dummies(df['Type'])
# Last Updated encoding

df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
# Convert kbytes to Mbytes 

k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()

converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))

df.loc[k_indices,'Size'] = converter
# Size cleaning

df['Size'] = df['Size'].apply(lambda x: x.strip('M'))

df[df['Size'] == 'Varies with device'] = 0

df['Size'] = df['Size'].astype(float)
# Split data into training and testing sets

features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']

features.extend(category_list)

X = df[features]

y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)
# Look at the 15 closest neighbors

model = KNeighborsRegressor(n_neighbors=15)
# Find the mean accuracy of knn regression using X_test and y_test

model.fit(X_train, y_train)
# Calculate the mean accuracy of the KNN model

accuracy = model.score(X_test,y_test)

'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
# Try different numbers of n_estimators - this will take a minute or so

n_neighbors = np.arange(1, 20, 1)

scores = []

for n in n_neighbors:

    model.set_params(n_neighbors=n)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

plt.figure(figsize=(7, 5))

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