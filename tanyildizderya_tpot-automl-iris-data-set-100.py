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
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
iris.data[0:5], iris.target
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=8, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print("Accuracy is {}%".format(tpot.score(X_test, y_test)*100))
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
extrTrClassfr = ExtraTreesClassifier(bootstrap=False, criterion="gini",max_features=0.6500000000000001,min_samples_leaf=18, min_samples_split=14, n_estimators=100)
extrTrClassfr_model = extrTrClassfr.fit(X_train, y_train)
extrTrClassfr_model
y_pred = extrTrClassfr_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
