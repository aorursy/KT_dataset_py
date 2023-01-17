# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test. csv")  

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, accuracy_score





# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Obtain target and predictors

y = X_full.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = X_full[features]

X_test = X_test_full[features]



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

                                                      
X_train.head()
# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)





for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d \t\t MAE: %d" % (i+1, mae))

    

    
# Fill in the best model

best_model = model_3



# Check your answer

step_1.check()
# Lines below will give you a hint or solution code

step_1.hint()

step_1.solution()
# Finding the best leaf node of model_3

leaf_nodes =  [5, 50, 500, 5000]



def leaf_node_func(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    model_3 = RandomForestRegressor(max_leaf_nodes = max_leaf_nodes, n_estimators=100, criterion='mae', random_state=0)

    model_3.fit(X_train, y_train)

    mod_3_pred = model_3.predict(X_valid)

    return mean_absolute_error(y_valid, mod_3_pred)





# Creating a loop for our function to iterate through leaf_nodes

for max_leaf_nodes in leaf_nodes:

    mae_2 = leaf_node_func(max_leaf_nodes, X_train, X_valid, y_train, y_valid)

    print("Leaf node:  %d \t\t Mean Absolute Error: %d"%(max_leaf_nodes, mae_2))

    

    
# Best leaf node 

candidate_max_leaf_nodes = [5, 50, 500, 5000]



# Dictionary is utilized

scores = {leaf_size: leaf_node_func(leaf_size, X_train, X_valid, y_train, y_valid) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

best_tree_size
# Defining a model...

my_model = RandomForestRegressor(n_estimators=100, criterion='mae', max_leaf_nodes=500, random_state=0)



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

step_2.hint()

step_2.solution()
# Fit the model to the training data

my_model.fit(X, y)



# Generate test pedictions

preds_test = my_model.predict(X_test)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)