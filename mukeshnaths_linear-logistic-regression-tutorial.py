# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import sklearn as sk

from sklearn.datasets import load_iris, load_boston

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris_data_array = load_iris(return_X_y = False)

boston_data_array = load_boston(return_X_y = False)
print('\n______________IRIS   DATASET______________\n')

print(iris_data_array.feature_names, iris_data_array.target_names, iris_data_array.DESCR)

#print(boston_data_array.feature_names, boston_data_array.target_names)

print("\n\n______________BOSTON   DATASET______________\n")

print(boston_data_array.feature_names, boston_data_array.DESCR, boston_data_array.target)

iris_data = pd.DataFrame(iris_data_array.data, columns = iris_data_array.feature_names)

iris_target = pd.DataFrame(iris_data_array.target, columns = ["Iris_Type"])

iris = pd.concat([iris_data,iris_target],axis=1,sort = False)

boston_data = pd.DataFrame(boston_data_array.data, columns = boston_data_array.feature_names)

boston_target = pd.DataFrame(boston_data_array.target, columns = ["Price"])

boston = pd.concat([boston_data,boston_target],axis=1,sort = False)
print(iris)

print(boston)

#boston_data_array.feature_names
logreg = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=126)



X, y = load_iris(return_X_y = True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 25)



logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



metrics.confusion_matrix(y_test,y_pred)
print(X,y,X_train, X_test, y_train, y_test,y_pred)
linreg = LinearRegression()



#X, y = boston.loc[:,['LSTAT','RM']], boston.loc[:,'Price']

X, y = boston.loc[:,['RM']], boston.loc[:,'Price']

#print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 25)



linreg.fit(X_train, y_train)



y_pred = linreg.predict(X_test)



r2 = metrics.r2_score(y_test, y_pred)

print(r2)
print(X,y,X_train, X_test, y_train, y_test,y_pred)
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_pred, color='blue', linewidth=0.5)
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_pred, color='blue', linewidth=0.5)
