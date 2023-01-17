import numpy as np

import pandas as pd

from xgboost import XGBClassifier
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train = train.dropna(axis = 'columns')
train1 = pd.concat([train, pd.get_dummies(train['LotShape'], prefix = 'LotShape'), pd.get_dummies(train['LandContour'], prefix = 'LandContour')], axis = 1)
predictor_cols = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd',

                  'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 

                  'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl']
train_y = train1.SalePrice

train_X = train1[predictor_cols]
model = XGBClassifier()
model.fit(train_X, train_y)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test = test.dropna(axis = 'columns')

test1 = pd.concat([test, pd.get_dummies(test['LotShape'], prefix = 'LotShape'), pd.get_dummies(test['LandContour'], prefix = 'LandContour')], axis = 1)
test_X = test1[predictor_cols]
predicted_prices = model.predict(test_X)
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test1.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission2-SVR_RuehleMax.csv', index=False)