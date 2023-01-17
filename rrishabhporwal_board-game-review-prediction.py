# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the Data

games = pd.read_csv('../input/games.csv')
# Print the names of the columns in games
print(games.columns)
print(games.shape)
# make a histogram of all the ratings in the average rating column
plt.hist(games['average_rating'])
plt.show()
# Print the row of all the games with zero scores
print(games[games['average_rating'] == 0].iloc[0])

# Print the first row of games with scores greater than 0
print(games[games['average_rating'] > 0].iloc[0])
# Remove any rows without user reviews
games = games[games['users_rated'] > 0]

# Remove any rows with missing values
games = games.dropna(axis=0)

# Make a histogram of all the average ratings
plt.hist(games["average_rating"])
plt.show()
print(games.columns)
#Correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax=8, square = True)
plt.show()
# Get all the columns from the dataframe
columns = games.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type", "id","name"]]

# Store the variable we'll be predicting on
target = "average_rating"
# Generating training and test datasets
from sklearn.model_selection import train_test_split

# Generate training set 
train = games.sample(frac=0.8, random_state = 1)

#Select anything not in the training set and put it in test
test = games.loc[~games.index.isin(train.index)]

# Print shapes
print(train.shape)
print(test.shape)
# Import Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression()

# Fit the model the training data
LR.fit(train[columns], train[target])
# Generate predictions for the test set
predictions = LR.predict(test[columns])

# Compute error between our test predictions and actual values
mean_squared_error(predictions, test[target])
# Import the random forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

# Fit to the data
RFR.fit(train[columns], train[target])
# make predictions
predictions = RFR.predict(test[columns])

# Compute the error between out test predictions and actual values
mean_squared_error(predictions, test[target])
test[columns].iloc[0]
# Make predictions with both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

# Print out the predictions
print(rating_LR)
print(rating_RFR)
test[target].iloc[0]
