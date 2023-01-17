# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")



#split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.ix[:,:'PetalWidthCm'], iris['Species'], random_state=0)



#set up xgboost with grid search and k-fold cv

from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier



param_grid = {

        'max_depth': [1, 2, 3, 4],

        'n_estimators': [4, 5, 6],

        'learning_rate': [0.001, 0.01, 0.1, 0.2, 1],

                }



clf = XGBClassifier()

grid_search = GridSearchCV(clf, param_grid, cv=5)

grid_search.fit(X_train, y_train)



print ('Best parameters:', grid_search.best_params_)

print ('Best estimator:', grid_search.best_estimator_)

print ('Best score:', grid_search.best_score_)