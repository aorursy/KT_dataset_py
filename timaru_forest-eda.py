# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the data

df_train = pd.read_csv('../input/learn-together/train.csv')

df_test = pd.read_csv('../input/learn-together/test.csv')
df_train.head()




print('Train size: ',df_train.shape)

print('Test size: ',df_test.shape)
#why is there only 55 columns in Test - whats different



train_cols = df_train.columns

test_cols = df_test.columns



missing_cols = train_cols.difference(test_cols)

print('Missing cols in df_test are -', missing_cols)
# What are the columns 



df_train.columns



# looking deeper



df_train.info()
# Create target object and call it y

y = df_train.Cover_Type



#Create X for the Features

#features = ['Elevation','Aspect','Slope']



features = df_test.columns.values.tolist()

print(features)



X = df_train[features]
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

forest_model = DecisionTreeRegressor(random_state=1)

# Fit Model

forest_model.fit(train_X, train_y)
# Make validation predictions and calculate mean absolute error

val_predictions = forest_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
# Using best value for max_leaf_nodes

forest_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

forest_model.fit(train_X, train_y)

val_predictions = forest_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)



# read test data file using pandas

test_data = df_test



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X  = test_data[features]



# make predictions which we will submit. 

test_preds = forest_model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)