import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfg

# Read the data
train = pd.read_csv('../input/train.csv')
train=train.dropna(axis=1)
y= train.SalePrice

#predictor_cols = ['MSZoning','LotArea','Condition1','Condition2','OverallQual','OverallCond','YearBuilt','ExterQual','Functional','GarageArea','SaleCondition']
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd','OverallCond','YrSold']

x=train[predictor_cols]


model=rfg()
model.fit(x,y)
test = pd.read_csv('../input/test.csv')

test_X = test[predictor_cols]

predicted_prices = model.predict(test_X)

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)