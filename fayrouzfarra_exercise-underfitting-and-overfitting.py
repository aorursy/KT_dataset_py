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

min_mae=get_mae(candidate_max_leaf_nodes[0],train_X,val_X,train_y,val_y)

results={}

beat_index=0

for i in range(0, len(candidate_max_leaf_nodes)):

    mae=get_mae(candidate_max_leaf_nodes[i],train_X,val_X,train_y,val_y)

    results[candidate_max_leaf_nodes[i]]=mae

    print('Max leaf nodes: {}, Validation mae: {:,.0f}'.format(candidate_max_leaf_nodes[i],mae))

    

    if min_mae>mae:

        min_mae=mae

        best_index=i

        

print('min mae: {:,.0f}, best tree size:{} '.format(min_mae, candidate_max_leaf_nodes[best_index]))

        

    
min(results,key=results.get)
l,l_mae=zip(*results.items())

l,l_mae
import matplotlib.pylab as plt

plt.figure(figsize=(10,6))

plt.plot(l,l_mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in candidate_max_leaf_nodes:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    #print("max_leaf_nodes :{}, Validation MAE: {:,.0f}".format(candidate_max_leaf_nodes[i],mae))

    

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))



# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = 100



# Check your answer

step_1.check()

for max_leaf_nodes in [50,60,70,80,90,100,110 ]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

for max_leaf_nodes in [60,61,62,63,64,65,66,67,68,69,70,71,72]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)



# Check your answer

step_2.check()
# step_2.hint()

# step_2.solution()
mae_dict={}



candidate_max_leaf_nodes= range(10,500,10)



for i in range(0, len(candidate_max_leaf_nodes)):

    mae=get_mae(candidate_max_leaf_nodes[i],train_X,val_X,train_y,val_y)

    mae_dict[candidate_max_leaf_nodes[i]]=mae

    print(candidate_max_leaf_nodes[i],"\t\t",mae)

    

min(mae_dict,key=mae_dict.get)
l,l_mae=zip(*mae_dict.items())

plt.plot(l,l_mae)