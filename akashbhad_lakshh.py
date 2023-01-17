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

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.getcwd())

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

SUSY = pd.read_csv("../input/trace-lhc/SUSY.csv",sep=',',header=None)
print("Dataset Lenght:: ", len(SUSY))

print("Dataset Shape:: ", SUSY.shape)
SUSY.head()
X = SUSY.values[:, 1:]

Y = SUSY.values[:,0]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.33, random_state = 200)
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

clf_gini.fit(X_train, y_train)

print("Accuracy score (training): {0:.3f}".format(clf_gini.score(X_train, y_train)))

print("Accuracy score (validation): {0:.3f}".format(clf_gini.score(X_test, y_test)))
clf_entropy = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=5, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=100, splitter='best')

clf_entropy.fit(X_train, y_train)

print("Accuracy score (training): {0:.3f}".format(clf_entropy.score(X_train, y_train)))

print("Accuracy score (validation): {0:.3f}".format(clf_entropy.score(X_test, y_test)))
#lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]



#for learning_rate in lr_list:

gb_clf = BaggingClassifier(base_estimator=clf_entropy,n_estimators=10,random_state=200)

gb_clf.fit(X_train, y_train)



#print("Learning rate: ", learning_rate)

print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))

print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
# Confusion Matrix:

gb_clf.fit(X_train, y_train)

print('Confusion Matrix:\n', confusion_matrix(y_test, gb_clf.predict(X_test)))

#

# Cross Validation based multiple metric evaluation:

nfolds = 10

def tn(y_true, y_pred): 

	return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred): 

	return confusion_matrix(y_true, y_pred)[0, 1]

def fn(y_true, y_pred): 

	return confusion_matrix(y_true, y_pred)[1, 0]

def tp(y_true, y_pred): 

	return confusion_matrix(y_true, y_pred)[1, 1]

#

scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),

           'fp': make_scorer(fp), 'fn': make_scorer(fn),

           'ac' : make_scorer(accuracy_score),

           're' : make_scorer(recall_score),

           'pr' : make_scorer(precision_score),

           'f1' : make_scorer(f1_score),

           'auc' : make_scorer(roc_auc_score),

          }           

#

cv_results = cross_validate(gb_clf, X_train, y_train, scoring=scoring, cv=StratifiedKFold(n_splits=nfolds, random_state=200))

# CV scores:

print('Cross Validation scores (nfolds = %d):'% nfolds)

print('tp: ', cv_results['test_tp'], '; mean:', cv_results['test_tp'].mean())

print('fn: ', cv_results['test_fn'], '; mean:', cv_results['test_fn'].mean())

print('fp: ', cv_results['test_fp'], '; mean:', cv_results['test_fp'].mean())

print('tn: ', cv_results['test_tn'], '; mean:', cv_results['test_tn'].mean())

print('ac: ', cv_results['test_ac'], '; mean:', cv_results['test_ac'].mean())

print('re: ', cv_results['test_re'], '; mean:', cv_results['test_re'].mean())

print('pr: ', cv_results['test_pr'], '; mean:', cv_results['test_pr'].mean())

print('f1: ', cv_results['test_f1'], '; mean:', cv_results['test_f1'].mean())

print('auc: ', cv_results['test_auc'], '; mean:', cv_results['test_auc'].mean())

y_pred = clf_gini.predict(X_test)

y_pred
y_pred_en = clf_entropy.predict(X_test)

y_pred_en
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)