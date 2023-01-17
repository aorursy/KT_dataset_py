import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
train_df.info()
test_df.info()
test_df = test_df.select_dtypes(include=['int64'])
train_df.fillna(value = train_df.mean())
plt.figure(figsize=(16,8))

sns.heatmap(train_df.corr())
test_df.columns
X = train_df[['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

       'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold']]
y = train_df[['SalePrice']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
simul_lm = LinearRegression()
simul_lm.fit(X_train, y_train)
simul_predict = simul_lm.predict(X_test)
plt.scatter(y_test, simul_predict)
from sklearn import metrics
print('MAE', metrics.mean_absolute_error(y_test,simul_predict))
print('MSE', metrics.mean_squared_error(y_test,simul_predict))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,simul_predict)))
lm = LinearRegression()
lm.fit(X, y)
print(lm.intercept_)
print(lm.coef_)
predict = lm.predict(test_df)
print(predict)
results = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission = pd.DataFrame()

submission['Id'] = results['Id']

submission['SalePrice'] = predict
submission.head()
submission.to_csv('submission.csv', index=False)