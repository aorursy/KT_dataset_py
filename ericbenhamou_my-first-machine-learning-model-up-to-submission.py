import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())
print(sorted(data.columns))
price_data = data.SalePrice
print(price_data.head())
columns_of_interest = ['TotRmsAbvGrd', 'TotRmsAbvGrd']
two_columns_of_data = data[columns_of_interest]
two_columns_of_data.describe()
y= data.SalePrice
predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
              'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictors]
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X,y)
print('Making predictions for the following 5 houses:')
print(X.head())
print("the predictions are")
print(model.predict(X.head()))
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = DecisionTreeRegressor()
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y, val_predictions))
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, train_y, test_X, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    return mean_absolute_error(test_y, y_pred)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae= get_mae(max_leaf_nodes, train_X, train_y, val_X, val_y)
    print( "Max leaf nodes: %d \\ Mean absolute Error %d" %(max_leaf_nodes, my_mae))

from sklearn.ensemble import RandomForestRegressor

for max_leaf_nodes in [5, 50, 500, 5000]:
    model2 = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes)
    model2.fit(train_X, train_y)
    y_preds = model2.predict(val_X)
    my_mae= mean_absolute_error(val_y, y_preds)
    print( "Max leaf nodes: %d \\ Mean absolute Error %d" %(max_leaf_nodes, my_mae))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

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
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)