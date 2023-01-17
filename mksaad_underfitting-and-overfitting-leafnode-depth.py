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

min_mae =  get_mae(candidate_max_leaf_nodes[0], train_X, val_X, train_y, val_y)

best_index = 0

results = {}

for i in range(1, len(candidate_max_leaf_nodes)):

    mae = get_mae(candidate_max_leaf_nodes[i], train_X, val_X, train_y, val_y)

    results[candidate_max_leaf_nodes[i]] = mae

    print("max_leaf_node: {}, Validation MAE: {:,.0f}".format(candidate_max_leaf_nodes[i], mae))

    if mae < min_mae:

        min_mae = mae

        best_index = i 

        



# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = candidate_max_leaf_nodes[best_index]

print('min mae', min_mae, 'best tree size', best_tree_size)

print('best tree size using dictionary', min(results, key=results.get))

step_1.check()
import matplotlib.pylab as plt

l, l_mae = zip(*results.items()) # unpack a list of pairs into two tuples

plt.figure(figsize=(20, 6))

plt.plot(l, l_mae)

plt.show()
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)

step_2.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# step_2.hint()

# step_2.solution()
candidate_max_leaf_nodes = list(range(10, 60, 5))

candidate_max_depth = list(range(2, 12))

print(candidate_max_leaf_nodes)

print(candidate_max_depth)

print(len(candidate_max_leaf_nodes), len(candidate_max_depth))
def get_mae_v2(max_leaf_nodes, max_depth, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

leaf_node_dict = {}

depth_dict = {}

for i in candidate_max_leaf_nodes:

    for j in candidate_max_depth:

        mae = get_mae_v2(i, j, train_X, val_X, train_y, val_y)

        leaf_node_dict[i] = mae 

        depth_dict[j] = mae 
min(leaf_node_dict, key=leaf_node_dict.get), min(depth_dict, key=depth_dict.get)
import matplotlib.pylab as plt

plt.figure(figsize=(20, 6))

l, l_mae = zip(*leaf_node_dict.items())

plt.subplot(2, 1, 1)

plt.plot(l, l_mae)

d, d_mae = zip(*depth_dict.items())

plt.subplot(2, 1, 2)

plt.plot(d, d_mae)

plt.show()