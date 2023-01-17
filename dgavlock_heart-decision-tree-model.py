# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.

 

heart_data = pd.read_csv("../input/heart.csv")

heart_data.head()
heart_data.describe()
heart_data.corr()
y = heart_data.target

features = ["cp", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

X = heart_data[features]



#splitting into training and validation datasets

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



#defining decision tree model

target_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=500)



#Fitting the model to training data



target_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = target_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)
# The MAE of the model

val_mae
#Defining get_mae for optimization loop

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# Creating Optimization loop

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}



# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = min(scores, key=scores.get)



print(best_tree_size)
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)



final_predict = final_model.predict(X)

mae = mean_absolute_error(final_predict,y)

print(mae)