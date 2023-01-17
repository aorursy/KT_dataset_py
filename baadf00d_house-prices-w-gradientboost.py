# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
sample = pd.read_csv('../input/sample_submission.csv')
test = pd.read_csv('../input/test.csv')
for i, t in zip(train.dtypes.index, train.dtypes):
    if t == np.object:
        train[i] = pd.factorize(train[i])[0]
        
for i, t in zip(test.dtypes.index, test.dtypes):
    if t == np.object:
        test[i] = pd.factorize(test[i])[0]
train.head()
from sklearn.model_selection import train_test_split

X, y = train.iloc[:, :-1], train.iloc[:, -1]
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(
    X.iloc[:, 1:], y
)
X_train.shape
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=256, random_state=0, learning_rate=0.2, max_depth=2)
model.fit(X_train, y_train)

print('Train accuracy : {:.2f}'.format(model.score(X_train, y_train)))
print('Test accuracy : {:.2f}'.format(model.score(X_test, y_test)))
test = test.fillna(test.mean())

res = pd.DataFrame({'Id':test.Id, 'SalePrice':model.predict(test.values[:, 1:])})
res.to_csv('submission.csv', index=False)