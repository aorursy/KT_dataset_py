
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
data = pd.read_csv("../input/menu.csv")
data.info()


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn , X = x_train, y=y_train, cv=10)

