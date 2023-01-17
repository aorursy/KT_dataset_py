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
import pandas as pd

from sklearn.model_selection import train_test_split



X=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

X_test_full = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")



X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)



low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64","float64"]]



my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_val = X_val_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



X_train=pd.get_dummies(X_train)

X_val=pd.get_dummies(X_val)

X_test=pd.get_dummies(X_test)

X_train, X_val  = X_train.align(X_val, join="left", axis=1)

X_train, X_test = X_train.align(X_test, join="left", axis=1)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



model_1 = XGBRegressor(random_state=0)

model_1.fit(X_train, y_train)

predictions_1 = model_1.predict(X_val)

mae_1 = mean_absolute_error(predictions_1, y_val)



print("Mean Absolute Error", mae_1)
model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05)

model_2.fit(X_train, y_train)

predictions_2 = model_2.predict(X_val)

mae_2 = mean_absolute_error(predictions_2, y_val)



print("Mean Absolute Error", mae_2)

preds_test = model_2.predict(X_test)
output = pd.DataFrame({'Id':X_test.index,

                       'SalePrice':preds_test})

output.to_csv('submission.csv', index=False)