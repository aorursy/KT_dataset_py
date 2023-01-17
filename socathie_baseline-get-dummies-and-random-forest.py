# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test
import matplotlib.pyplot as plt
y = train.SalePrice.values

_ = plt.hist(y)

y.shape
train = train.drop("SalePrice", axis=1)

train
df = pd.concat([train, test], axis=0)

df
df = pd.get_dummies(df)

df = df.fillna(-1)

df
train = df.iloc[:len(train)]

train
test = df.iloc[:len(test)]

test
X = train.iloc[:,1:].values

X.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor(random_state=0)



params = {'n_estimators': [100,1000],

    'max_features':['log2', 'auto', 'sqrt'],

    'max_depth': [2,10,100],

}



grid = GridSearchCV(estimator=rf,

                       param_grid=params,

                       scoring='neg_mean_squared_error',

                       cv=3,

                       verbose=1,

                       n_jobs=-1)



grid.fit(X, y)



best = grid.best_estimator_
y_train = best.predict(X)

mean_squared_error(np.log(y), np.log(y_train), squared=False)
y_pred = best.predict(test.iloc[:,1:].values)
sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

sub
for i in range(len(y_pred)):

    sub.loc[i,"SalePrice"] = y_pred[i]

sub.to_csv("submission.csv", index=False)

sub