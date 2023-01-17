# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



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
train_file_path  = '../input/train.csv'

train_data = pd.read_csv(train_file_path)



# cols_with_missing = [col for col in train_data.columns 

#                                  if train_data[col].isnull().any()]                                  

# candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



# low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

#                                 candidate_train_predictors[cname].nunique() < 10 and

#                                 candidate_train_predictors[cname].dtype == "object"]

# numeric_cols = [cname for cname in candidate_train_predictors.columns if 

#                                 candidate_train_predictors[cname].dtype in ['int64', 'float64']]

# my_cols = low_cardinality_cols + numeric_cols





# train_predictors = candidate_train_predictors[my_cols]



















# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=31)





# Drop houses where the target is missing

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)



y = train_data.SalePrice



# Since missing values isn't the focus of this tutorial, we use the simplest

# possible approach, which drops these columns. 

# For more detail (and a better approach) to missing values, see

# https://www.kaggle.com/dansbecker/handling-missing-values

cols_with_missing = [col for col in train_data.columns 

                                 if train_data[col].isnull().any()]                                  

candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)



# "cardinality" means the number of unique values in a column.

# We use it as our only way to select categorical columns here. This is convenient, though

# a little arbitrary.

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[common_cols]





#one_hot_encoding for handling categorical data

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)



X = one_hot_encoded_training_predictors





# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)





# predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])







# # create test_X which comes from test_data but includes only the columns you used for prediction.

# # The list of columns is stored in a variable called features



cols_with_missing_test  = [col for col in test_data.columns 

                                 if test_data[col].isnull().any()]                       

candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing_test, axis=1)





low_cardinality_cols_test = [c_name for c_name in candidate_test_predictors.columns if 

                                candidate_test_predictors[c_name].nunique() < 10 and

                                candidate_test_predictors[c_name].dtype == "object"]

numeric_cols_test = [c_name for c_name in candidate_test_predictors.columns if 

                                candidate_test_predictors[c_name].dtype in ['int64', 'float64']]



my_cols_test = low_cardinality_cols_test + numeric_cols_test

test_predictors = candidate_test_predictors[common_cols]

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

test_X = one_hot_encoded_test_predictors



# common_cols = train_data.intersection(test_cols)

# train_not_test = train_cols.difference(test_cols)





# # make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)







# # The lines below shows how to save predictions in format used for competition scoring

# # Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
len(one_hot_encoded_training_predictors.columns)



print(test_predictors.columns.intersection(train_predictors.columns))

#train_predictors

# NB !!!!

#THIS CODE IS EXECUTED BEFORE RUNNING TO FIT THE MODEL. THE VARIABLE common_cols IS USED TO DECIDE WHICH 

#COLUMNS WILL FIT TO THE TRAIN MODEL REGARDING TO ITS INTERSECTION WITH THE TEST DATA SO IT WONT BE ERROR





# common_cols = numeric_cols_test.intersection(numeric_cols) 



# print(common_cols)

# train_not_test = candidate_train_predictors.columns.difference(candidate_test_predictors.columns)

# print (train_not_test)

# print (len(candidate_train_predictors.columns))

# print (len(candidate_test_predictors.columns))





final_train = train_data[numeric_cols]

final_test = test_data[numeric_cols_test]





common_cols = final_test.columns.intersection(final_train.columns)



test_data.isnull()
# Check your answer

step_1.check()

step_1.solution()