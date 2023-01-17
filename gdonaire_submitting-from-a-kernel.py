import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# Read the data
train = pd.read_csv('../input/train.csv')
train.describe()
train.columns

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
#predictor_cols = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd'] --> 0.0066
#predictor_cols = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd'] --> 0.0050
# MSLE = 0.0054
#predictor_cols = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd']
# MSLE = 0.0049
#predictor_cols = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'GrLivArea', 'GarageArea']
predictor_cols = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF','BedroomAbvGr']
# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

train_y_predict = my_model.predict(train_X)
msle = mean_squared_log_error(train_y, train_y_predict)
print("Mean Squared Log Error: ", msle)
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