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
from sklearn.datasets import load_boston

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston["MEDV"] = boston_dataset.target
boston.head()
corr = boston.corr()
corr.style.background_gradient(cmap="coolwarm").set_precision(2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston[["LSTAT"]], boston[["MEDV"]], test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
reg = LinearRegression().fit(X_train,y_train)
print(reg.score(X_train,y_train))
#print(reg.coef_)
#print(reg.intercept_)
y_pred = reg.predict(X_test)
y_train = reg.predict(X_train)
print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(y_pred)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
X = boston[['LSTAT']].values
y = boston[['MEDV']].values
regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(),X.max(),1)[:,np.newaxis]
regr = regr.fit(X,y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y,regr.predict(X))

regr = regr.fit(X_quad,y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y,regr.predict(X_quad))

regr = regr.fit(X_cubic,y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y,regr.predict(X_cubic))

plt.scatter(X,y,label='training points',color='lightgray')
plt.plot(X_fit,y_lin_fit,label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit,y_quad_fit,label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit,y_cubic_fit,label='cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the popelation [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()
from sklearn.tree import DecisionTreeRegressor
X = boston[["LSTAT"]] 
y = np.array(boston[["MEDV"]])
    
from sklearn import tree
clf = tree.DecisionTreeRegressor(max_leaf_nodes=15)
clf = clf.fit(X, y)
tree_r2 = r2_score(y, clf.predict(X))
tree_r2
from sklearn.tree import DecisionTreeRegressor
X = boston[["LSTAT",'RM']] 
y = np.array(boston[["MEDV"]])
    
from sklearn import tree
clf = tree.DecisionTreeRegressor(max_leaf_nodes=15)
clf = clf.fit(X, y)
tree_r2 = r2_score(y, clf.predict(X))
tree_r2
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon = 1.5)
svm_reg.fit(X,y)
from sklearn.svm import SVR
X = boston.drop('MEDV',axis = 1)
y = np.array(boston[["MEDV"]])
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X,y)
y_svm_fit = svm_poly_reg.predict(X)
svm_r2 = svm_poly_reg.score(X,y)
svm_r2