import sys

import logging



import numpy as np

import scipy as sp

import sklearn

import pandas as pd

import statsmodels.api as sm

from statsmodels.formula.api import ols



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import seaborn as sn

sn.set_context("poster")

sn.set(rc={'figure.figsize': (16, 9.)})

sn.set_style("whitegrid")



import pandas as pd

pd.set_option("display.max_rows", 120)

pd.set_option("display.max_columns", 120)
columns_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data_set = pd.read_csv('../input/housing.csv',names=columns_names,delim_whitespace=True,)
data_set.head()
data_set.hist(figsize=(15,10),grid=False)

plt.show()
data_set.describe()
plt.figure(figsize=(15,10)) 

sn.heatmap(data_set.corr(),annot=True) 
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_df = scaler.fit_transform(data_set)

scaled_df = pd.DataFrame(scaled_df,columns=columns_names) 
X = scaled_df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]

y = scaled_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
def pretty_print_linear(coefs, names = None, sort = False):

    if names == None:

        names = ["X%s" % x for x in range(len(coefs))]

    lst = zip(coefs, names)

    if sort:

        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))

    return " + ".join("%s * %s" % (round(coef, 3), name)

                                   for coef, name in lst)
linear_regression = LinearRegression()

model = linear_regression.fit(X_train, y_train)

print('The accuracy of the Linear Regression is: {:.2f}'.format(model.score(X_train,y_train)))

print('The accuracy of the Linear Regression is: {:.2f}'.format(model.score(X_test,y_test)))

pretty_print_linear(model.coef_)
ridge = Ridge()

model = ridge.fit(X_train, y_train)

print('The accuracy of the Ridge Regression is: {:.2f}'.format(model.score(X_train,y_train)))

print('The accuracy of the Ridge Regression is: {:.2f}'.format(model.score(X_test,y_test)))

pretty_print_linear(model.coef_)
lasso = Lasso()

model = lasso.fit(X_train, y_train)

print('The accuracy of the Lasso Regression is: {:.2f}'.format(model.score(X_train,y_train)))

print('The accuracy of the Lasso Regression is: {:.2f}'.format(model.score(X_test,y_test)))

pretty_print_linear(model.coef_)
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor().fit(X_train, y_train)

print('The accuracy of the Decision Tree Regression is: {:.2f}'.format(model.score(X_train,y_train)))

print('The accuracy of the Decision Tree Regression is: {:.2f}'.format(model.score(X_test,y_test)))
n_features = X.shape[1]

plt.barh(range(n_features), model.feature_importances_, align = 'center')

plt.yticks(np.arange(n_features), X.columns)

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.ylim(-1, n_features)