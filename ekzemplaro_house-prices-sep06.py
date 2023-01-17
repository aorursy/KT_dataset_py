

import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestRegressor
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')



print(train_df.shape)

print(test_df.shape)

print(submission.shape)

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']



train_x = train_df[predictor_cols]

train_y = train_df.SalePrice





model = RandomForestRegressor()

model.fit(train_x, train_y)



test_x = test_df[predictor_cols]



pred = model.predict(test_x)



print(pred[:20])
submission.SalePrice = pred

print(submission.head(20))

file_submit = "house_prices_submit.csv"

#

submission.to_csv(file_submit,index=False)