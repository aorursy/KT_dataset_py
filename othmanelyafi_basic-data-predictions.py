import numpy as np

import pandas as pd



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor



# Read the training data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()

# Read the test data

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test.head()
# pull data into target (y) and predictors (X)

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data

train_X = train[predictor_cols]



# Create test predictors data

test_X = test[predictor_cols]

model = XGBClassifier()

model.fit(train_X, train_y)

print(model)
pred=model.predict(test_X)

print(pred)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})



submission.to_csv('submission.csv', index=False)