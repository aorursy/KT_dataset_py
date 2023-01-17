# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
y = pd.read_csv('../input/all_train.csv')
X = y.drop('has_parkinson', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y['has_parkinson'].values, train_size=0.75, test_size=0.25, random_state=16)
# Score on the training set was:0.766783625731
exported_pipeline = make_pipeline(
    Binarizer(threshold=0.5),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.4, n_estimators=100), step=0.15),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.4, min_samples_leaf=9, min_samples_split=17, n_estimators=100)),
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.95, n_estimators=100), threshold=0.4),
    DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_leaf=13, min_samples_split=9)
)
exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, results)))
