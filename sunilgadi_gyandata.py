# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cars_sample.csv', encoding = 'unicode_escape')
df.head(10)
#to remove more than 4 NA's in a row
df['no_nan'] = df.apply(lambda x: df.shape[1]-x.count(), axis=1)
df = df[df.no_nan <=3]
df.shape
#replace NaN's
df['vehicleType'].fillna('others',inplace = True)
df['gearbox'].fillna(df['gearbox'].mode()[0],inplace = True)
df['model'].fillna('others',inplace = True)
df['fuelType'].fillna(df['fuelType'].mode()[0],inplace = True)
df['notRepairedDamage'].fillna('na',inplace = True)
df['Year_Created'] = df['dateCreated'].apply(lambda x: re.split(r"[/ ]+", x)[2])
df['Age'] = pd.to_numeric(df['Year_Created']) - pd.to_numeric(df['yearOfRegistration'])
df_1 = df[['price','vehicleType','gearbox','powerPS','model','kilometer','fuelType','brand', 'notRepairedDamage', 'Age']]
#outlier removal
df_1 = df_1[(df_1['price']<= 13000) & (df_1['price'] >= 400) ]
df_1 = df_1[(df_1['Age']<= 30) & (df_1['Age'] >= 0) ]
df_1 = df_1[(df_1['powerPS']<= 800) & (df_1['powerPS'] >= 40) ]
df_1.shape
x = plt.boxplot(df_1['price'])
[item.get_ydata() for item in x['whiskers']]
df_2 = pd.get_dummies(df_1, columns=['vehicleType', 'gearbox' , 'model' , 'fuelType' , 'brand', 'notRepairedDamage'], drop_first=True)
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_2.iloc[:,1:], df_2.iloc[:,0], test_size = 0.25, random_state = 127)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
from sklearn.linear_model import Ridge
linear_regressor = Ridge(alpha=5.0)
linear_regressor.fit(train_features, train_labels) 
predictions = linear_regressor.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
from sklearn.metrics import r2_score
r2_score(test_labels, predictions)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(train_features, train_labels)
predictions = regressor.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
from sklearn.metrics import r2_score
r2_score(test_labels, predictions)
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000,random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
from sklearn.metrics import r2_score
r2_score(test_labels, predictions)
feature_list = list(df_1.iloc[:,1:].columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]