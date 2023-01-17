# Code you have previously used to load data

import pandas as pd

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]



# Specify Model

iowa_model = DecisionTreeRegressor()

# Fit Model

iowa_model.fit(X, y)



print("First in-sample predictions:", iowa_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex4 import *

print("Setup Complete")
home_data
X
y
# Import the train_test_split function and uncomment

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y,test_size= 0.2, random_state = 0)



from sklearn.metrics import mean_absolute_error

melbourne_model = DecisionTreeRegressor()

# Fit model

melbourne_model.fit(train_X, train_y)





train_y
val_y
val_predictions
val_predictions.size
# print the top few validation predictions

print(val_predictions[:5])

# print the top few actual prices from validation data

print(val_y[1:5])
from sklearn.metrics import mean_absolute_error





val_predictions = melbourne_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
