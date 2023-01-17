import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# save filepath to variable for easier access
melbourne_file_path = '../input/train.csv'
# read the data and store the data in a DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of Melbourne data
print(melbourne_data.describe())
# print columns of Melbourne data
print(melbourne_data.columns)
# store the series of prices separately as melbourne_price_data.
melbourne_saleprice_data = melbourne_data.SalePrice
# use head command to return top few lines of code
print(melbourne_saleprice_data.head())
# selecting multiple columns
columns_of_interest = ['LotArea', 'YearBuilt']
two_columns_of_data = melbourne_data[columns_of_interest]
two_columns_of_data.describe()
# Get Sale price
y = melbourne_data.SalePrice
melbourne_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 
                        'TotRmsAbvGrd']
X = melbourne_data[melbourne_predictors]
from sklearn.tree import DecisionTreeRegressor

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)
print("Making predictionsfor the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time 
# the script is run.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))





import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read in data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'TotRmsAbvGrd' ]

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)





my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# use any filename. 
my_submission.to_csv('prices.csv', index=False)
print(melbourne_data.isnull().sum())