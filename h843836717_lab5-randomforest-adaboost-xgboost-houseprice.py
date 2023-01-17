# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1400, max_depth=5,max_leaf_nodes=200, n_jobs=-1)

rfr.fit(X,y)

rfr_y_pred = rfr.predict(X_test)
from math import sqrt

sqrt(mean_squared_error(y_test, rfr_y_pred))
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.1)

ada.fit(X_train, y_train)

ada_y_pred=ada.predict(X_test)
from math import sqrt

sqrt(mean_squared_error(y_test, ada_y_pred))
from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train, y_train)

xgb_y_pred = xgb.predict(X_test)
from math import sqrt

sqrt(mean_squared_error(y_test, xgb_y_pred))
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_X=test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']]

test_y=test_X.values.reshape(-1,5)

predicted_price=rfr.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

my_submission.to_csv('submission_Zihao_Han.csv', index=False)
