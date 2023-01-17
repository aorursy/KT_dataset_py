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



'''

/kaggle/input/house-prices-advanced-regression-techniques/train.csv

/kaggle/input/house-prices-advanced-regression-techniques/test.csv

/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv

/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt

'''



import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



target = '/kaggle/input/house-prices-advanced-regression-techniques/'



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = target + 'train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

#rf_model_on_full_data = RandomForestRegressor(random_state=1)

#rf_model_on_full_data = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

rf_model_on_full_data  = RandomForestRegressor(n_estimators=100, random_state=0)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)
# path to file you will use for predictions

test_data_path = target + 'test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# testing mae scores



model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error



# train_X, val_X, train_y, val_y

# Function for comparing different models

def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))