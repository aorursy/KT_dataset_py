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
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PolynomialFeatures
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression()), 
])
X_train = train[['LotFrontage','LotArea']].fillna(0)
X_test  = test[['LotFrontage','LotArea']].fillna(0)
y_train = train['SalePrice']
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
scores = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
np.mean(pow(-scores, 0.5))
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
sub = pd.DataFrame({'Id': test.Id, 'SalePrice': preds})
sub.to_csv('submission.csv', index=False)
