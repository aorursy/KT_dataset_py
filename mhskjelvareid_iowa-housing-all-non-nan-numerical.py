# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read. 

iowa_file_path = '../input/train.csv'



# Read data

home_data = pd.read_csv(iowa_file_path)
# List of all columns

all_feat = (home_data.columns.tolist())



# List of numerical data columns

num_data = home_data._get_numeric_data()

num_feat = (num_data.columns.tolist())



# List of NaN data columns

nan_feat = home_data.columns[home_data.isnull().sum()>0].tolist()



# Filter

features = [all_feat[i] for i in range(len(all_feat)) if (all_feat[i] in num_feat) ] 

features = [features[i] for i in range(len(features)) if not(features[i] in nan_feat)]



features.remove('SalePrice')  # Don't include y in X

print(features)

# Define X and y

X = home_data.loc[:,features]

y = home_data.SalePrice
# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(X, y)
# Path to file for predictions

test_data_path = '../input/test.csv'



# Read test data file using pandas

test_data = pd.read_csv(test_data_path)



# Create test_X which comes from test_data but includes only numeric, non-NaN features

test_X = test_data[features]

test_X.fillna(0,inplace=True) # "Hack" to fill NaNs which are in columns different from the training set. Not happy with this one. 



# make predictions 

test_preds = rf_model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)