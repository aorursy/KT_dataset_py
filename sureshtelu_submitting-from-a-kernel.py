import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read the data
train = pd.read_csv('../input/train.csv')
#train = train.dropna(subset=['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd'])
#train = train.dropna(axis=1)
#cols_with_missing = [col for col in train.columns 
#                                 if train[col].isnull().any()]
#train = train.drop(cols_with_missing, axis=1)
 
#train = train.drop(train[train.SalePrice < 100000].index)
#train = train.drop(train[train.SalePrice > 300000].index)
#train = train.drop(train[train.LotArea < 5000 ].index)
#train = train.drop(train[train.LotArea > 100000 ].index)
#train = train.drop(train[train.YearBuilt < 1950 ].index)
# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','LotShape','PoolArea','Fence','MSZoning']

# Create training predictors data
train_X = train[predictor_cols]
print(train.head())
my_model = LogisticRegression()
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