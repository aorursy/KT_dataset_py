import numpy as np
import pandas as pd
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.corr()
train.corr()['SalePrice'].abs().sort_values(ascending=False).head(6)
train.shape
import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(15,10))
sns.regplot(x='OverallQual' , y='SalePrice' , data=train )
plt.figure(figsize=(15,10))
sns.regplot(x='GrLivArea' , y='SalePrice' , data=train )
plt.figure(figsize=(15,10))
sns.regplot(x='GarageCars' , y='SalePrice' , data=train )
test['GarageCars'].isnull().sum()
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean() )
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean() )
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['TotalBsmtSF'].isnull().sum()
x_train = train[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']].values
x_test = test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']].values
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
y_train = train['SalePrice'].values
from sklearn.model_selection import train_test_split

x_train_1 , x_train_test , y_train_1, y_train_test = train_test_split(x_train,y_train
                                                                     ,test_size=1/5
                                                                     ,random_state=0)

x_train_1.shape
x_train_test.shape
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train_1 , y_train_1)
y_train_pred = regressor.predict(x_train_test)
np.mean(np.absolute(y_train_pred - y_train_test))
newRegressor = LinearRegression()
newRegressor.fit(x_train,y_train)
y_pred = newRegressor.predict(x_train)
np.mean(np.absolute(y_pred - y_train))
test['Id'].values
prediction = newRegressor.predict(x_test)
submission = pd.DataFrame({'Id':test['Id'].values,'SalePrice':prediction})
submission.head()
req_submission = submission.set_index('Id')
req_submission.head()
req_submission.to_csv('house_price_prediction_2.csv')
