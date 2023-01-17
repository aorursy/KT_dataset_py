import pandas as pd
from xgboost import XGBRegressor
train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
#These two DataFrames will be used to train the model.
train_X = train.loc[:,:'SaleCondition'] #'SaleCondition' is the second last column before 'SalePrice'
train_y = train.loc[:,'SalePrice']

#This DataFrame will be used for the predictions
#we will submit.
test_X = test.copy()
numeric_cols = train_X.dtypes[train_X.dtypes != 'object'].index
train_X = train_X[numeric_cols]

test_X = test_X[numeric_cols]
train_X = train_X.fillna(train_X.mean())

test_X = test_X.fillna(test_X.mean())
model = XGBRegressor()

model.fit(train_X, train_y, verbose=False)
predictions = model.predict(test_X)
#Create a DataFrame for our submission
submission = pd.DataFrame({'Id':test_X.index, 'SalePrice':predictions})
#Write the submission to a 'csv' file
submission.to_csv('submission.csv', index=False)