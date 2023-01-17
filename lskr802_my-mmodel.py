import numpy as np

import pandas as pd

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv')

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

train_X = train[predictor_cols]



my_model = RandomForestRegressor()

my_model.fit(train_X,train_y)
test = pd.read_csv('../input/test.csv')

test_X = test[predictor_cols]

predicted_price = my_model.predict(test_X)

print(predicted_price)
submission = pd.DataFrame({'Id':test.Id,'SalePrice':predicted_price})

submission.to_csv('submission.csv',index = False)