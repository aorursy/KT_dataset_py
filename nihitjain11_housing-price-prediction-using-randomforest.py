import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv')

train_y = train.SalePrice
predictors = ['LotArea','OverallQual','YearBuilt','TotRmsAbvGrd']
train_x = train[predictors]

forest_model = RandomForestRegressor()
forest_model.fit(train_x,train_y)
test = pd.read_csv('../input/test.csv')

test_x = test[predictors]
forest_predict = forest_model.predict(test_x)
print (forest_predict)
submission = pd.DataFrame({'Id':test.Id,'SalePrice':forest_predict})
print(submission.head())
submission.to_csv('submission.csv',index=False)