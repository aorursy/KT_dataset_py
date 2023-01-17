# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

SUSY = pd.read_csv("../input/trace-lhc/SUSY.csv",sep=',',header=None)
print("Dataset Lenght:: ", len(SUSY))

print("Dataset Shape:: ", SUSY.shape)
SUSY.head()
X = SUSY.values[:, 1:14]

Y = SUSY.values[:,0]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.4, random_state = 19)
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

clf_gini.fit(X_train, y_train)
clf_entropy = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=5, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

clf_entropy.fit(X_train, y_train)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]



for learning_rate in lr_list:

    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)

    gb_clf.fit(X_train, y_train)



    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)

gb_clf2.fit(X_train, y_train)

predictions = gb_clf2.predict(X_test)



print("Confusion Matrix:")

print(confusion_matrix(y_test, predictions))



print("Classification Report")

print(classification_report(y_test, predictions))
y_pred = clf_gini.predict(X_test)

y_pred
y_pred_en = clf_entropy.predict(X_test)

y_pred_en
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
xgb_clf = XGBClassifier()

xgb_clf.fit(X_train, y_train)
score = xgb_clf.score(X_test, y_test)

print(score)