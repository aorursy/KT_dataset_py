# Code you have previously used to load data

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor





# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

Y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_Y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_Y)

print("Validation MAE: {:,.0f}".format(val_mae))



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex5 import *

print("\nSetup complete")
def get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_Y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_Y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

"""

candidates_mae = []

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf in candidate_max_leaf_nodes:

    candidates_mae.append(get_mae(max_leaf, train_X, val_X, train_Y, val_Y))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

min_mae_index = candidates_mae.index(min(candidates_mae))

best_tree_size = candidate_max_leaf_nodes[min_mae_index]



print(candidate_max_leaf_nodes)

print(candidates_mae)

"""



candidates_leaf_mae = {max_leaf: get_mae(max_leaf, train_X, val_X, train_Y, val_Y) for max_leaf in candidate_max_leaf_nodes}

best_tree_size = min(candidates_leaf_mae, key=candidates_leaf_mae.get)

print("Best tree size:", best_tree_size)



step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)



final_model.fit(X, Y)



step_2.check()
# step_2.hint()

# step_2.solution()