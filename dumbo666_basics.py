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
df = pd.read_csv('../input/train.csv')

df.head(n=10)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LassoCV, LinearRegression
X_train, X_test, y_train, y_test = train_test_split(

    normalize(df[['ra', 'dec', 'u', 'g', 'r', 'i', 'z','size', 'ellipticity']]), df['redshift'],

test_size=0.33, random_state=42)



print(X_train.shape, y_train.shape)



tr = DecisionTreeRegressor()

lasso = LassoCV(eps=0.1, cv=5)

lin_reg = LinearRegression()
reg = tr.fit(X_train, y_train)

las = lasso.fit(X_train, y_train)

lin = lin_reg.fit(X_train, y_train)
tree_pred = reg.predict(X_test)

lasso_pred = las.predict(X_test)

lin_pred = lin.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Decision tree MSE : ', mean_squared_error(tree_pred, y_test))
print('LassoCV MSE : ', mean_squared_error(lasso_pred, y_test))
print('LinReg MSE : ', mean_squared_error(lin_pred, y_test))
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(42,),

                   batch_size=256, verbose=True,

                   max_iter=100, solver='sgd',

                   early_stopping=True).fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
print('MultiLayerPerceptron MSE : ', mean_squared_error(mlp_pred, y_test))
import xgboost as xgb
xgb_model = xgb.XGBRegressor(max_depth=6)
xgb_reg = xgb_model.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)

print('XGBoost MSE : ', mean_squared_error(xgb_pred, y_test))
test = pd.read_csv('../input/test.csv')

test.head(n=10)
test_pred_with_xgb = xgb_reg.predict(

    normalize(test[['ra', 'dec', 'u', 'g', 'r', 'i', 'z','size', 'ellipticity']]))
test['id'].values, test_pred_with_xgb
predictions = pd.DataFrame(data=test_pred_with_xgb, index=range(test_pred_with_xgb.size),

                           columns=['redshift'])

predictions.head(n=10)
predictions.to_csv('predictions.csv', index_label='id', columns=['redshift'])
output = pd.read_csv('predictions.csv')