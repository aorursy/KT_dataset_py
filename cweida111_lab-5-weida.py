# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
corrdf = df.corr()

corrdf
#Find our most effective columns

pos = 0

columns = corrdf.columns

train_columns = []

for i in corrdf.iloc[:37,37]:

    if i >= 0.2:

        print(i, " ", columns[pos])

        

        train_columns.append(columns[pos])

    pos += 1

print(train_columns)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
X = df[train_columns]

y = df[['SalePrice']]

X.head()
columns = X.columns



i = 0

to_drop = []

while i < len(X.columns):

    if X.iloc[:,i].dtype == object:

        X = pd.concat([X,pd.get_dummies(X.iloc[:,i], prefix=columns[i])],axis=1)

        to_drop.append(columns[i])

        #X.drop(columns[i],axis=1, inplace=True)

        

    i+=1

for i in range(len(to_drop)):

    X.drop(to_drop[i],axis=1, inplace=True)
X = X.fillna(0)
X.head()

from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)



rnd_reg = RandomForestRegressor(n_estimators = 500, 

max_leaf_nodes = 1500, n_jobs = -1)



rnd_reg.fit(X_train, y_train.values.ravel())



y_pred1 = rnd_reg.predict(X_test)
print("RandomForestRegressor RMSE: ",sqrt(mean_squared_error(y_test, y_pred1)))
err = sqrt(mean_squared_error(y_test, y_pred1))

while err > 24000:

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    rnd_reg.fit(X_train, y_train.values.ravel())

    y_pred1 = rnd_reg.predict(X_test)

    err = sqrt(mean_squared_error(y_test, y_pred1))

print(err)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
Adb = AdaBoostRegressor()

Dtr = DecisionTreeRegressor()
ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=1000, learning_rate=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

ada_reg.fit(X_train, y_train.values.ravel())

y_pred_ada=ada_reg.predict(X_test)


print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))
from xgboost import XGBRegressor
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
while err > 23000:

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    xgb_reg.fit(X_train, y_train.values.ravel())

    y_pred_xgb = xgb_reg.predict(X_test)

    err = sqrt(mean_squared_error(y_test, y_pred_xgb))

print(err)
test_X = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_X = test_X[train_columns]

test_X.columns
test_y = test_X.values.reshape(-1,)

predicted_price = xgb_reg.predict(test_X)

predicted_price
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

my_submission.to_csv('submissionXGBoost.csv', index=False)