import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import os





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        

#Stores githib-hosted link in variable (for information about the data, please refer to the README document)



train_data_url = '../input/home-data-for-ml-course/train.csv'



test_data_url = '../input/home-data-for-ml-course/test.csv'



#train_data is the data used to train data 

train_data = pd.read_csv(train_data_url)



#test_data is the data used to test the model's performance

test_data = pd.read_csv(test_data_url)

        

predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train_data[predictors]

y = train_data.SalePrice

        



def train_model(model_arg, X_arg, y_arg):

  """

  Performs a train_test_split function to split the data into a training set and validation set.

  As the names suggest, inputted model trains on the training set, and then the function returns

  a list of the predictions for the validation y-values (target) and a list of the input validation

  x-values (predictors).



  train_model(my_model, predictors, target_variable)

  >>> returns my_model.predict(predictors), target_variable

  """

  train_X, val_X, train_y, val_y = train_test_split(X_arg, y_arg, random_state=1)



  model_arg.fit(train_X, train_y)

  return model_arg.predict(val_X), val_y



def test_model(model_arg, X_arg, y_arg, X_test_arg):

  """

  Same function as train_model except for the absence of a train_test_split function and no actual

  y-value output like val_y in train_model. Used to fit an optimized ML model with the entire

  predictors data before predicting with the test predictors dataset.

  """

  model_arg.fit(X_arg, y_arg)

  return model_arg.predict(X_test_arg)



def mae(model_arg, X_arg, y_arg):

  """

  Performs the same function as the mean_absolute_error with the addition of a

  model argument that incoporates the train_model function to return the

  mean absolute error (MAE).



  mae(my_model, predictors, target_variable)

  >>> returns mean_absolute_error(pred_y, val_y)

  """

  pred_y, val_y = train_model(model_arg, X_arg, y_arg)

  mae = mean_absolute_error(pred_y, val_y)

  return mae



def initialize_DT(max_leaf_nodes_arg):

  """

  Creates a Decision Tree Regressor model with the specified max_leaf_nodes.



  initialize_DT(200)

  >>> DecisionTreeRegressor(max_leaf_nodes=200,random_state=1)

  """

  DT_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes_arg, random_state=1)

  return DT_model



def experiment_with_DT(max_leaf_nodes_range_arg, X_arg, y_arg, is_data_table = False):

  """

  This model incorporates initialize_DT(), mae(), train_model() function to perform

  one of two actions: (1) a list of MAE values (if is_data_table = False or there is

  no specification) or (2) a dictionary with the keys as max_leaf_nodes (i.e. ints

  like 2,25,126) and values as the MAE values. Note that max_leaf_nodes_range_arg

  CANNOT be less than 2. The function finally returns the variable mae_results which

  is either intialized as a dictionary or list depending on teh is_data_table Boolean.



  experiment_with_DT(30, predictors, target, is_data_table = False)

  >>> returns mae_results = [MAE @ 2 max_leaf_nodes],...,[MAE @ 30 max_leaf_nodes]



  experiment_with_DT(30, predictors, target)

  >>> returns mae_results = [MAE @ 2 max_leaf_nodes],...,[MAE @ 30 max_leaf_nodes]



  experiment_with_DT(30, predictors, target, is_data_table = True)

  >>> returns mae_results = {2:MAE @ 2 max_leaf_nodes,...,30:MAE @ 30 max_leaf_nodes}

  """

  if is_data_table:

    mae_results = {}

  else:

    mae_results = []



  max_leaf_nodes_range = list(range(2,max_leaf_nodes_range_arg+1))

  

  for max_leaf_nodes in max_leaf_nodes_range:

    DT_model = initialize_DT(max_leaf_nodes)

    if is_data_table:

      mae_results[max_leaf_nodes] = mae(DT_model, X_arg, y_arg)

    else:

      mae_results.append(mae(DT_model, X_arg, y_arg))

  

  return mae_results



def optimize_DT(max_leaf_nodes_range_arg, X_arg, y_arg):

  """

  Further extends experiment_with_DT by finding the lowest mean absolute error 

  (best for the ML model) and then returns the corresponding max_leaf_node

  to the lowest MAE as a tuple such that (optimal max_leaf_node, lowest MAE).



  optimize_DT(200, predictors, target)

  >>> (71, 20123.142)

  """

  mae_results = experiment_with_DT(max_leaf_nodes_range_arg, X_arg, y_arg)

                                   

  min_mae = min(mae_results)

  optimal_leaf_node = mae_results.index(min_mae) + 2



  return (optimal_leaf_node, round(min_mae, 3))





#Finds the optimal max_leaf_nodes value for Decision Tree and stores both the optimal value and minimized MAE

optimal_max_leaf_node, optimal_mae = optimize_DT(200,X,y)

#Creates a dictionary filled with max_leaf_nodes values from 2 to 200 and corresponding MAE values

max_leaf_node_data_table = experiment_with_DT(200,X,y, is_data_table=True)

#Seperates the dictionary into two lists

(max_leaf_nodes_x_values, mae_y_values) = zip(*max_leaf_node_data_table.items())



optimized_DT = initialize_DT(optimal_max_leaf_node)



#predictors is a group of columns defined in section "Defining Predictors"

X_testing = test_data[predictors]



predicted_y = test_model(optimized_DT, X, y, X_testing)



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': predicted_y})



output.to_csv('submission.csv',index=False)