import pandas as pd



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head(10)

train.columns
train.dtypes
data_types = zip(list(train.columns), list(train.dtypes))

print(list(data_types))
train2 = train[['Id','MSSubClass','MSSubClass','LotArea','OverallQual','YearBuilt','YearRemodAdd','MiscVal','MoSold','YrSold','SalePrice']].dropna()

test2 = test[['Id','MSSubClass','MSSubClass','LotArea','OverallQual','YearBuilt','YearRemodAdd','MiscVal','MoSold','YrSold']].dropna()



X_train = train2[['MSSubClass','MSSubClass','LotArea','OverallQual','YearBuilt','YearRemodAdd','MiscVal','MoSold','YrSold']]



X_test = test2[['MSSubClass','MSSubClass','LotArea','OverallQual','YearBuilt','YearRemodAdd','MiscVal','MoSold','YrSold']]



y_train = train2['SalePrice']
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor().fit(X_train, y_train)

predictions = gbr.predict(X_test)

test2['SalePrice'] = predictions



result = test2[['Id','SalePrice']]



import os

os.chdir(r'/kaggle/working')

result.to_csv(r'result.csv')
