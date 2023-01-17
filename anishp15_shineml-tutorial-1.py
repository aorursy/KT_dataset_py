# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

 

# Any results you write to the current directory are saved as output.
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

print(home_data)

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",

                      "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]



X=home_data[feature_names]

y = home_data["SalePrice"]

print(X)

from sklearn.tree import DecisionTreeRegressor

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = DecisionTreeRegressor(random_state=1)



# Fit the model

iowa_model.fit(X, y)





predictions = iowa_model.predict(X)

print(predictions)

# Import the train_test_split function and uncomment

from sklearn.model_selection import train_test_split



# fill in and uncomment

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify the model

iowa_model2 = DecisionTreeRegressor(random_state=1)



iowa_model2.fit(train_X, train_y)

# Fit iowa_model with the training data.

val_predictions = iowa_model.predict(val_X)



print(val_predictions)



from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)



# uncomment following line to see the validation_mae

print(val_mae)
