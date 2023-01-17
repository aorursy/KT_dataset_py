# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
print(train.shape)
print(test.shape)
train.describe()
print(train.SalePrice.skew())
plt.hist(train.SalePrice, color = "red")
plt.show()
target = np.log(train.SalePrice)
print(target.skew())
plt.hist(target, color='green')
plt.show()
corr = train.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]
train.OverallQual.unique()
plt.scatter(x=train['GrLivArea'], y=target)
plt.show()
plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))
plt.show()
plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice))
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice))
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
one_hot_encoded_train = pd.get_dummies(train)
one_hot_encoded_train.head()
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
data.head()
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.22)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
from xgboost import XGBRegressor

xgb_model = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
xgb_model.fit(X_train, y_train, verbose=False)

model = xgb_model.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.5).fit(X_train,y_train)
print("R2 score from test set :"+str(lasso.score(X_test,y_test)))

predictions = lasso.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = lasso.predict(feats)
final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions

submission.to_csv('submission2.csv', index=False)