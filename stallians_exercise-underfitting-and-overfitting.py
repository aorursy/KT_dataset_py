# Code you have previously used to load data

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor





# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



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

print("Validation MAE: {:,.0f}".format(val_mae))



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex5 import *

print("\nSetup complete")
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

best_leaf_node=-1

min_mae=100000

for leaf_node in candidate_max_leaf_nodes:

    mae = get_mae(leaf_node, train_X, val_X, train_y, val_y)

    if mae < min_mae:

        min_mae = mae

        best_leaf_node = leaf_node



# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = best_leaf_node



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

step_1.solution()
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)



# Check your answer

step_2.check()
# step_2.hint()

step_2.solution()