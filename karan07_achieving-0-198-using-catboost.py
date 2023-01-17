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
data=pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.fillna(-999, inplace=True)
test.fillna(-999,inplace=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=67)
categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = model.predict(test)
submission.to_csv("Submission.csv", index=False)
