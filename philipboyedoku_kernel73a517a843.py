# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Path of the file to read

iowa_file_path = '../input/covid19gobaldataset/train_data.csv'





home_data = pd.read_csv(iowa_file_path)

#print(home_data.columns)

# Create target object and call it y

# Y= confirmed,death,recovery

y = home_data.confirmed

# Create X

features =[

       'days_since_pandemic', 'temperature','average_contact_tracing', 

       'no_of_gov_interventions', 'urban_population', 'household_size',

        'road_transport_expenditure', 'air_transport_expenditure',

       'train_transport_expenditure', 'elderly_population',

       ]

 

X = home_data[features]

#print(X)

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







forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

val_mae = mean_absolute_error(val_y, melb_preds)

print("Validation MAE for RandomForestRegressor: {:,.0f}".format(val_mae))



#predicting Ghana in for the next 4 weeks

iowa_file_path_ghana = '../input/ghana-test-dataset/ghana_test.csv'





home_data_ghana = pd.read_csv(iowa_file_path_ghana)

#print(home_data_ghana)

#print(home_data.columns)

# Create target object and call it y

y = home_data_ghana.confirmed

# Create X

features =[ 

       'days_since_pandemic', 'temperature','average_contact_tracing',

       'no_of_gov_interventions', 'urban_population', 'household_size',

        'road_transport_expenditure', 'air_transport_expenditure',

       'train_transport_expenditure', 'elderly_population',

       ]

 

X = home_data_ghana[features]

ghana_preds = forest_model.predict(X)

print(X.days_since_pandemic)

print(ghana_preds)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex6 import *

print("\nSetup complete")
