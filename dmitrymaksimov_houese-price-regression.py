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
data_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data_train.head()
X_train = data_train.drop(['Id','SalePrice'], axis=1)

y_train = data_train['SalePrice']

X_train
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

X_test.drop(['Id'],axis=1, inplace=True)

X_test
X_train.dropna().shape, X_test.dropna().shape
data_train[data_train.columns[:-1]].corrwith(data_train[data_train.columns[-1]])
data_train.mean()
X = X_train.append(X_test, ignore_index=True)
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']

numerical_columns   = [c for c in X.columns if X[c].dtype.name != 'object']
X[numerical_columns].dropna().shape, X[categorical_columns].dropna().shape, 
X_real = X[numerical_columns].fillna(0)
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer as DV

X_cat = X[categorical_columns].astype(str)

encoder = DV(sparse = False)

X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

scaler = StandardScaler()

X_real_scaled = scaler.fit_transform(X_real)

X_scaled = np.hstack((X_real_scaled[:len(y_train)], X_cat_oh[:len(y_train)]))
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression

cross_val_score(LinearRegression(), X_scaled, y_train, scoring='neg_mean_absolute_error', cv=3)
from sklearn.linear_model import Lasso, Ridge

regressor_L = Lasso()

regressor_R = Ridge()

print(cross_val_score(regressor_L, X_scaled, y_train, scoring='neg_mean_absolute_error', cv=3))

print(cross_val_score(regressor_R, X_scaled, y_train, scoring='neg_mean_absolute_error', cv=3))
regressor_L.get_params

params = {'alpha' : np.array([10**i for i in range(-3, 10)])}

grid_cv = GridSearchCV(regressor_L, params,scoring='neg_mean_absolute_error', cv=3)

grid_cv.fit(X_scaled, y_train)

print(grid_cv.best_estimator_, grid_cv.best_score_)
regressor_R.get_params

params = {'alpha' : np.array([10**i for i in range(-3, 10)])}

grid_cv_R = GridSearchCV(regressor_R, params,scoring='neg_mean_absolute_error', cv=3)

grid_cv_R.fit(X_scaled, y_train)

print(grid_cv_R.best_estimator_, grid_cv_R.best_score_)
from sklearn.ensemble import RandomForestRegressor

cross_val_score(RandomForestRegressor(), X_scaled, y_train, cv=3, scoring='neg_mean_absolute_error')

from xgboost import XGBRegressor

cross_val_score(XGBRegressor(), X_scaled, y_train, cv=3, scoring='neg_mean_absolute_error')
cross_val_score(Lasso(alpha=100), X_scaled, y_train, cv=3, scoring='neg_mean_absolute_error')
best_regressor = Lasso(alpha=100)

best_regressor.fit(X_scaled, y_train)

X_test_scaled = np.hstack((X_real_scaled[len(y_train):], X_cat_oh[len(y_train):]))

predict = best_regressor.predict(X_test_scaled)
index = [i for i in range(1461, 1461 + len(predict))]

d = {'id': index, 'SalePrice': predict}

output = pd.DataFrame(d)
output


output.to_csv("../working/submission.csv",index=False)