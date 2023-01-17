# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

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
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

def get_mae_RF(max_leaf_nodes, estimator, s_split, s_leaf, train_X, val_X, train_y, val_y):
    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, 
                                     n_estimators=estimator,  
                                     min_samples_split = s_split, 
                                     min_samples_leaf = s_leaf,
                                     random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    #print(rf_val_mae)
    return rf_val_mae
    
import numpy as np
candidate_max_leaf_nodes = [5,25,50,100,250,500]
n_estimators = [5,10,15,20,25,30,35,40,50]
min_samples_split = [2,3,5,7,9]
min_samples_leaf = [2,3,5,7,9]

# Loop to find the best leaft node candidate
result_acc = []
parameter_list = []
for leaf in candidate_max_leaf_nodes:
    for estimator in n_estimators:
        for s_split in min_samples_split:
            for s_leaf in min_samples_leaf:
                result_acc.append([get_mae_RF(leaf, estimator, s_split, s_leaf,
                                             train_X, val_X, train_y, val_y)])
                parameter_list.append([leaf,estimator,s_split, s_leaf])
result_acc = np.array(result_acc)
#print(result_acc)

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_parameters = parameter_list[result_acc.argmin()]
print(best_parameters)
rf_val_mae = get_mae_RF(best_parameters[0], best_parameters[1], best_parameters[2], best_parameters[3], 
                        train_X, val_X, train_y, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1, 
                                              max_leaf_nodes=best_parameters[0],
                                              n_estimators=best_parameters[1])

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(X,y)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)