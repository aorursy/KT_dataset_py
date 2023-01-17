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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
df = load_boston()

df.keys()
dataset = pd.DataFrame(df.data)

dataset.head()
dataset.columns = df.feature_names
dataset.head()
df.target.shape
dataset['Price'] = df.target
dataset.head()
X = dataset.iloc[:,:-1] ## Independent Features

y = dataset.iloc[:,-1] ## Dependent Features

print('-'*40)

print("X")

print('-'*40)

print(X)

print('-'*40)

print("Y")

print('-'*40)

print(y)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

lin_regressor = LinearRegression()

mse = cross_val_score(lin_regressor, X, y, scoring='neg_mean_squared_error',cv=5)

mean_mse = np.mean(mse)

print(mse)

print(mean_mse)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ridge = Ridge()

parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso = Lasso()

parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
prediction_lasso=lasso_regressor.predict(X_test)

prediction_ridge = ridge_regressor.predict(X_test)
import seaborn as sns

sns.distplot(y_test-prediction_lasso)
import seaborn as sns

sns.distplot(y_test-prediction_ridge)