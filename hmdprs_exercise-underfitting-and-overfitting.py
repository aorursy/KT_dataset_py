# load data

import pandas as pd

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)



# create target object, y

y = home_data['SalePrice']

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# split X and y into validation and training data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# specify model

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)

# fit Model

iowa_model.fit(train_X, train_y)



# make validation predictions and calculate mean absolute error (mae)

from sklearn.metrics import mean_absolute_error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print(f"Validation MAE: {val_mae:,.0f}")



# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex5 import *

print("\nSetup complete")
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    val_predictions = model.predict(val_X)

    mae = mean_absolute_error(val_y, val_predictions)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]



# write loop to find the ideal tree size from candidate_max_leaf_nodes



# my long solution :-(

acc_dict = dict()

for max_leaf_nodes in candidate_max_leaf_nodes:

    mae_now = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    acc_dict[max_leaf_nodes] = mae_now



acc_dict_keys = list(acc_dict.keys())

acc_dict_values = list(acc_dict.values())

min_value = min(acc_dict_values)



# store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = acc_dict_keys[ acc_dict_values.index(min_value) ]



# kaggle's short solution with a dict comprehension :-)

# scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# best_tree_size = min(scores, key=scores.get)



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)



# Check your answer

step_2.check()
# step_2.hint()

# step_2.solution()