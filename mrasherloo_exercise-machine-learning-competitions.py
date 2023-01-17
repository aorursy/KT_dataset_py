import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



iowa_file_path = '../input/home-data-for-ml-course/train.csv'



train = pd.read_csv(iowa_file_path)

y = train.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train[features]



test_data_path = '../input/home-data-for-ml-course/test.csv'

test = pd.read_csv(test_data_path)

test_X = test[features]
rf_model_on_full_data = RandomForestRegressor()

rf_model_on_full_data.fit(X, y)

y_preds = rf_model_on_full_data.predict(val_X)

print(mean_absolute_error(y_preds, val_y))
# path to file you will use for predictions



# read test data file using pandas



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)