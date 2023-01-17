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

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

df.head()

df = df.drop(['Serial No.'], axis=1)

df.isnull().sum()

from sklearn.model_selection import train_test_split



X = df.drop(['Chance of Admit '], axis=1)

y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error











regressor=Lasso()

regressor.fit(X_train, y_train)

predictions =regressor.predict(X_test)

print('Linear Regression', (np.sqrt(mean_squared_error(y_test, predictions))))
