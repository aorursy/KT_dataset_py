# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#our Avocado File 

avo_file_path ="../input/avocado.csv"
avo_data = pd.read_csv(avo_file_path)

#A check to see what the columns are
avo_data.columns

# dropna drops missing values (think of na as "not available")
avo_data = avo_data.dropna(axis=0)


# Any results you write to the current directory are saved as output.
#Prediction Target
y = avo_data.AveragePrice

#Features

avo_features = ['Total Volume','Total Bags','Small Bags','Large Bags','year']

X = avo_data[avo_features]

#Review


X.head()


X.describe()
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
avo_model = DecisionTreeRegressor(random_state=1)

# Fit model
avo_model.fit(X, y)
print(X.head())
print("The predictions are")
print(avo_model.predict(X.head()))

print(avo_model.predict(X))
from sklearn.metrics import mean_absolute_error

predicted_avo_prices = avo_model.predict(X)
print("The predicted price: ",predicted_avo_prices)
mean_absolute_error(y, predicted_avo_prices)
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
#avo_model = DecisionTreeRegressor()
# Fit model
avo_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = avo_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#22 cents off