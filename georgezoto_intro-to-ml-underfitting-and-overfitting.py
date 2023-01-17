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
iowa_model = DecisionTreeRegressor(max_depth=3, random_state=1) 
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))
print("Tree Model Depth: {}".format(iowa_model.get_depth()))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
iowa_model.get_depth()
from sklearn import tree

text_representation = tree.export_text(iowa_model)

#The text representation of this tree is pretty long so let's just show a few nodes instead
print(text_representation[:5000])
from matplotlib import pyplot as plt

#This will take a couple of minutes and use all your CPU
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(iowa_model, feature_names=features, filled=True)
#Very important step and package used in Step 4
!pip install graphviz --upgrade
import graphviz
print(graphviz.__version__)

#Scroll up and down, left and right to navigate this good looking tree :) 
graphviz_tree = tree.export_graphviz(iowa_model, out_file=None, 
                                feature_names=features,
                                filled=True)
graphviz.Source(graphviz_tree, format="png") 
!pip install dtreeviz #--use-feature=2020-resolver
from dtreeviz.trees import dtreeviz # remember to load the package

viz = dtreeviz(iowa_model, X, y,
                target_name="Price",
                feature_names=features)
viz
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    print('max_leaf_nodes: {:>3}, mae:{:<10}'.format(max_leaf_nodes, mae) )
    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
mae_max_leaf_nodes = [get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) for max_leaf_nodes in candidate_max_leaf_nodes]

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = candidate_max_leaf_nodes[mae_max_leaf_nodes.index(min(mae_max_leaf_nodes))]

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()
plt.figure(figsize=(12,7))
#plt.style.available[:5]
plt.style.use('fivethirtyeight')
plt.xlabel('Tree Depth')
plt.ylabel('MAE')
#plt.ylim([0, max(mae_max_leaf_nodes)])
plt.plot(candidate_max_leaf_nodes, mae_max_leaf_nodes);
plt.annotate("Min MAE:{:,.0f}, depth:{}".format(min(mae_max_leaf_nodes), best_tree_size), (best_tree_size, min(mae_max_leaf_nodes)+1000))
plt.show()
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
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

# Check your answer
step_2.check()
#step_2.hint()
#step_2.solution()