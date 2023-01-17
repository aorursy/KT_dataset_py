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
#import

import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

import matplotlib.pyplot as plt
#Data

train = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

X_train = train.iloc[:, :1]

y_train = train.iloc[:, 1:]

X_test = test.iloc[:, :1]

y_test = test.iloc[:, 1:]
#Imputer

ImputedModule = SimpleImputer(missing_values= np.nan, strategy= 'mean')

X_train = ImputedModule.fit_transform(X_train)

y_train = ImputedModule.fit_transform(y_train)

X_test = ImputedModule.fit_transform(X_test)

y_test = ImputedModule.fit_transform(y_test)
X_graph = X_test
#Polynomail Features

poly_reg = PolynomialFeatures()

X_train = poly_reg.fit_transform(X_train)

X_test = poly_reg.fit_transform(X_test)
#Model

LinearRegressionModel = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

LinearRegressionModel.fit(X_train, y_train)



print("Linear Regression Train Score: ",LinearRegressionModel.score(X_train, y_train))

print("Linear Regression Test Score: ",LinearRegressionModel.score(X_test, y_test ))
#Predict

y_pred = LinearRegressionModel.predict(X_test)
#Metrices

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')

print('Mean Absolute Error: ', MAEValue)



MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')

print('Mean Squared Error: ', MSEValue)



MdSEValue = median_absolute_error(y_test, y_pred, multioutput='uniform_average')

print('Median Absolute Error: ', MdSEValue)
#Graph

plt.style.use('seaborn-whitegrid')

ax = plt.axes()

ax.set(xlabel='X_train', ylabel='Y_Train and Y_Predict', title='Relationship between Y_Test and Y_Predict')

plt.scatter(X_graph,y_test, color='#2ecc71', alpha=.5, label='Test')

plt.plot(X_graph,y_pred, color='#f1c40f', label='Predict')

plt.legend(prop={'size': 16})