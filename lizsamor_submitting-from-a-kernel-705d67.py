import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# read the data
train = pd.read_csv('../input/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
# read test data
test = pd.read_csv('../input/test.csv')
# treat the test data in the same way as training data. In this case, pull same columns
test_X = test[predictor_cols]
# use the model to make predictions
predicted_prices = my_model.predict(test_X)
# we will look at the predicted prices to ensure we have something sensible
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('LizS_submission.csv', index=False)