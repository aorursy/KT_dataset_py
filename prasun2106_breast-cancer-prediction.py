# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
cancer.DESCR
data =  pd.DataFrame(cancer.data,columns = cancer.feature_names )
data.head()
cancer.target
data.head()
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
X = data
y = pd.DataFrame(cancer.target)
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y)
model_svm = SVC()
model_svm.fit(X_train, y_train)
prediction = model_svm.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, prediction)
metrics.classification_report(y_test, prediction)

metrics.confusion_matrix(y_test, prediction)
from sklearn.model_selection import GridSearchCV
param_grid = {"C":[0.001,0.1, 1, 10,100], "gamma": [1, 0.1,0.001,.0001]}
grid_search = GridSearchCV(SVC(), param_grid, verbose = 1)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
prediction_grid = grid_search.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test, prediction_grid)
metrics.confusion_matrix(y_test, prediction_grid)
metrics.classification_report(y_test, prediction_grid)