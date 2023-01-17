# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Forked from the exercise. I'll say "forked" to simplify later on when necessary.
# import pandas as pd # already initiated above
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Forked partially
# Path of the file to read
ks_data_path = '../input/ks-projects-201801.csv'
ks_data = pd.read_csv(ks_data_path)

# Filtering to only show successful projects
ks_data = ks_data[ks_data.state == 'successful']
ks_data = ks_data[ks_data.currency == 'USD']

# Forked partially
# Create X
features = ['goal', 'pledged', 'backers']
X = ks_data[features]
# Drops rows with missing values
#X = X.dropna(axis=0)
# Create target object and call it y
y = X.pledged

# Split into validation and training data, random_state set to 0 to ensure frozen randomness
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#BREAKKKKK=========================================

# Describe
X.describe()
def get_test_mae_lifting(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# This function returns the robust value of leaf nodes for get_test_mae_lifting()
def get_optimal_size(train_X, val_X, train_Y, val_Y):
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500] #This remain hardcoded for now.
    best_tree_size = candidate_max_leaf_nodes[0] #this code is self-made by me
    best_mae = get_test_mae_lifting(best_tree_size, train_X, val_X, train_y, val_y)
    compare_mae = best_mae

    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    for max_leaf in candidate_max_leaf_nodes[1:]:
        compare_mae = get_test_mae_lifting(max_leaf, train_X, val_X, train_y, val_y)
        if (best_mae >= compare_mae):
            best_mae = compare_mae
            best_tree_size = max_leaf
        print("Max leaf nodes: %d  \t Mean Absolute Error:  %d" %(max_leaf, compare_mae))
    return best_tree_size

# Initializing model
full_model = DecisionTreeRegressor(random_state=1)
train_model = DecisionTreeRegressor(random_state=1)
test_model = DecisionTreeRegressor(random_state=1)
best_size = int(get_optimal_size(train_X, val_X, train_y, val_y))
print("Best leaf nodes size is : %d" %(best_size))

# Calculating full MAE
full_model.fit(X, y)
full_predict = full_model.predict(X)
full_mae = mean_absolute_error(y, full_predict)
print("Full MAE : \t\t%d \n" %(full_mae), "Predictions : ", full_predict)

# Calculating train MAE
train_model.fit(train_X, train_y)
train_predict = train_model.predict(train_X)
train_mae = mean_absolute_error(train_y, train_predict)
print("Train MAE : \t\t%d \n" %(train_mae), "Predictions : ", train_predict)

# Calculating test MAE
test_model.fit(train_X, train_y)
test_predict = test_model.predict(val_X)
test_mae = mean_absolute_error(val_y, test_predict)
print("Test MAE : \t\t%d \n" %(test_mae), "Predictions : ", test_predict)

# Calculating robust test MAE
robust_model = DecisionTreeRegressor(max_leaf_nodes=best_size, random_state=1)

#robust_model.fit(train_X, train_y)
#robust_predict = robust_model.predict(val_X)
robust_mae = get_test_mae_lifting(best_size, train_X, val_X, train_y, val_y) #mean_absolute_error(val_y, robust_predict)
print("Robust Test MAE : \t%d \n" %(robust_mae), "Predictions : ", robust_predict)
# A slight modification for range() because range() only accepts integer.
def frange(start, stop, step):
    i = start
    while i <= stop:
        yield i
        i += step

# Ratio increment step amount
step = 0.1
# Range we want to test
ratio_range = frange(0.1, 0.9, step)

# A function to get MAE instantly, I'm getting more meh,with random_state = 1
def get_test_mae(train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# A function that prints all the possible MAE with given range.
def get_test_mae_ranged_ratio(data, ratio_range):
    for ratio in ratio_range:
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=ratio, random_state=1)
        print("Ratio = %f \tMAE = %d" %(ratio, get_test_mae(train_X, val_X, train_y, val_y)))
    
# Call for the function
get_test_mae_ranged_ratio(ks_data, ratio_range)
# Importing required libraries
import matplotlib.pyplot as plt
#import numpy as np # Already imported above.

# Deciding the maximum leaf number
max_leaf_arr = range(100, 5000, 100)

# Reinitialize the training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# A function that returns an array consisting of MAEs on every maximum leaf number array elements
def get_lifting_mae(max_leaf_arr, train_X, val_X, train_y, val_y):
    out = []
    candidate_max_leaf_nodes = max_leaf_arr
    for max_leaf in candidate_max_leaf_nodes:
        out.append(int(get_test_mae_lifting(max_leaf, train_X, val_X, train_y, val_y)))
        #print ("ping")
    return out

# Array variable
mae_yield = get_lifting_mae(max_leaf_arr, train_X, val_X, train_y, val_y)

#print (mae_yield)

# Setting and rinting the plot
plt.plot(max_leaf_arr, mae_yield, 'ro')
plt.xlabel('Maximum Leaf Size')
plt.ylabel('Mean Absolute Error')
plt.show()