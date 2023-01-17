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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dia = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')

print(dia.shape)

print(dia.isnull().sum())
dia.head()
dia.describe()
from sklearn.model_selection import train_test_split



# bagging ensembles

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



# Random Forests

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve



# Ada Boost Classifier

from sklearn.ensemble import AdaBoostClassifier



X = dia.drop('Outcome', axis = 1)

y = dia.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



# bagging ensembles

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state = 42), n_estimators = 500,

                            max_samples = 100, bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(X_train, y_train)



# determine accuracy score for the bagging method

bag_y_pred = bag_clf.predict(X_test)

bag_score = accuracy_score(y_test, bag_y_pred)



# standard decision tree classifier

tree_clf = DecisionTreeClassifier(random_state = 42)

tree_clf.fit(X_train, y_train)



# determine accuracy score for the Decision Tree method

tree_y_pred = tree_clf.predict(X_test)

tree_score = accuracy_score(y_test, tree_y_pred)



# Random Forests

rf_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1, random_state = 42)

rf_clf.fit(X_train, y_train)



# determine accuracy score for Random Forest method

rf_y_pred = rf_clf.predict(X_test)

rf_score = accuracy_score(y_test, rf_y_pred)



# Ada Boost Classifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 200, 

                             algorithm = "SAMME.R", learning_rate = 0.5, random_state = 42)

ada_clf.fit(X_train, y_train)



# determine accuracy score for Ada Boost Classifier

ada_y_pred = ada_clf.predict(X_test)

ada_score = accuracy_score(y_test, ada_y_pred)



# summary for all the results

print('Accuracy Score Results')

print('Bagging Classifier {:.2f}'.format(bag_score))

print('Decision Tree Classifier {:.2f}'.format(tree_score))

print('Random Forest {:.2f}'.format(rf_score))

print('Ada Boost Classifier {:.2f}'.format(ada_score))
# determine accuracy score for all methods

bag_y_prob = bag_clf.predict_proba(X_test)

bag_y_score = bag_y_prob[:, 1]

bag_fpr, bag_tpr, bag_threshold = roc_curve(y_test, bag_y_score)



tree_y_prob = tree_clf.predict_proba(X_test)

tree_y_score = tree_y_prob[:, 1]

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_test, tree_y_score)



rf_y_prob = rf_clf.predict_proba(X_test)

rf_y_score = rf_y_prob[:, 1]

rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, rf_y_score)



ada_y_prob = ada_clf.predict_proba(X_test)

ada_y_score = ada_y_prob[:, 1]

ada_fpr, ada_tpr, ada_threshold = roc_curve(y_test, ada_y_score)



# plotting ROC Curve to visualize all method

sns.set_style('whitegrid')

plt.figure(figsize = (10, 8))

plt.plot(bag_fpr, bag_tpr, label = 'Bagging Classifier')

plt.plot(tree_fpr, tree_tpr, label = 'Decision Tree Classifier')

plt.plot(rf_fpr, rf_tpr, label = 'Random Forest Classifier')

plt.plot(ada_fpr, ada_tpr, label = 'Ada Boost Classifier')



plt.plot([0, 1], [0, 1], color = 'blue', linestyle = '--')

plt.axis([0, 1, 0, 1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve', fontsize = 15)

plt.legend(loc = "lower right")

plt.show()