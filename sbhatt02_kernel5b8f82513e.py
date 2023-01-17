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
input_data = pd.read_csv("/kaggle/input/the-consulting-club-analytics-challenge/Train.csv")

inputtest_data = pd.read_csv("/kaggle/input/the-consulting-club-analytics-challenge/Test.csv")
print(input_data.isna().any())
input_data = input_data.fillna(0)
X, y = input_data.iloc[:,1:-1],input_data.iloc[:,-1]

X = pd.get_dummies(X)

X
X_test = inputtest_data.iloc[:,1:]

X_test = pd.get_dummies(X_test)

X_test
X_test=X_test.reindex(columns = X.columns, fill_value=0)

import xgboost as xgb

data_dmatrix = xgb.DMatrix(data=X,label=y)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X,y)
prediction= xg_reg.predict(X_test)
prediction
submission = pd.DataFrame({ 'Id': inputtest_data['Id'], 'price_usd': prediction })
submission.to_csv("submission.csv", index=False)