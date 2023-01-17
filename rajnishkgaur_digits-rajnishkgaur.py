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
X= pd.read_csv('../input/train.csv')
X.head()
y=X['label']
x=X.drop(labels=['label'], axis=1)
X_train, X_test, y_train, y_test = x[:34000], x[34000:], y[:34000], y[34000:]
test_set=pd.read_csv('../input/test.csv')
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3,
                                    method="predict_proba")
results_rf=forest_clf.predict(test_set)
results_rf = pd.Series(results_rf, name="Label")

submission_rf = pd.concat(
    [pd.Series(range(1, len(test_set) + 1), name="ImageId"), results_rf], axis=1)

submission_rf.to_csv("results_rf.csv", index=False)
!ls
