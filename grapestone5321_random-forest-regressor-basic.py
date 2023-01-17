import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

train_X = train[predictor_cols]
model = RandomForestRegressor()

model.fit(train_X, train_y)
test = pd.read_csv('../input/test.csv')

test.head()
test_X = test[predictor_cols]



predicted_prices = model.predict(test_X)

print(predicted_prices)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

submission.to_csv('submission.csv', index=False)