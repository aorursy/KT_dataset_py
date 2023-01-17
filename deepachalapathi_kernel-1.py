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
dataset = pd.read_csv('../input/insurance/insurance.csv')

dataset.head(5)
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
#print(X[0])
#print(y)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 1] = le.fit_transform(X[:, 1])

X[:, 4] = le.fit_transform(X[:, 4])
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X[0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()

regressor.fit(X_poly, y_train)

'''

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 34, max_depth =4,  random_state = 0, verbose = 1)

regressor.fit(X_train, y_train)

#obtained parameters form grid search

#criterion': 'mse', 'max_depth': 4, 'n_estimators': 34, 'oob_score': True

'''

from sklearn.svm import SVR

regressor = SVR(kernel = 'linear')

regressor.fit(X_train, y_train)

'''

'''

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

'''
'''

y_pred = regressor.predict(poly_reg.transform(X_test))

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

'''

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
#k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
'''

from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [i for i in range (10,51)], 

               'criterion': ['mse','mae'],

                'max_depth': [i for i in range(11)],

               'oob_score':[True,False]}]

grid_search = GridSearchCV(estimator = regressor,

                           param_grid = parameters,

                           scoring = 'r2',

                           cv = 10,

                           n_jobs = -1)

grid_search.fit(X_train, y_train)

#best_accuracy = grid_search.best_score_

#best_parameters = grid_search.best_params_

print("Best Score", grid_search.best_score_)

print("Best Parameters:", grid_search.best_params_)

#print("Best estimator : ", grid_search.best_estimator_)

'''