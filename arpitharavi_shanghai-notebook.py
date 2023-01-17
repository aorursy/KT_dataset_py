# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection



print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.linear_model import LinearRegression

# Any results you write to the current directory are saved as output.
shanghai = pd.read_csv('../input/shanghaiData.csv')
print(shanghai.dtypes)
X = shanghai[['alumni','award','hici','ns','pub','pcp']]

y = shanghai.total_score

X = X.fillna(method='ffill')

y = y.fillna(method='ffill')

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
reg = LinearRegression()

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
print('Scoring =',reg.score(X_train,y_train))

print('Coefficient :',reg.coef_)

print('Intercept :',reg.intercept_)
print('Root Mean Square Error =',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Mean Absolute Error =',metrics.mean_absolute_error(y_test, y_pred))

print('Mean Square Error =',metrics.mean_squared_error(y_test, y_pred))

print('R^2 =',metrics.r2_score(y_test, y_pred))
kfold = model_selection.KFold(n_splits=10, random_state=7)

scoring = 'neg_mean_absolute_error'

results = cross_val_score(reg, X, y, cv=kfold, scoring=scoring)

print("Mean Absolute Error: %.3f (%.3f)" % (results.mean(), results.std()))
scoring = 'neg_mean_squared_error'

results = model_selection.cross_val_score(reg, X, y, cv=kfold, scoring=scoring)

print("Mean Squared Error: %.3f (%.3f)" % (results.mean(), results.std()))
scoring = 'r2'

results = model_selection.cross_val_score(reg, X, y, cv=kfold, scoring=scoring)

print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))