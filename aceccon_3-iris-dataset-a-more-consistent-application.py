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
#Reading data from CSV file

df = pd.read_csv("../input/Iris.csv")#Defining data and label
#Defining data and label

X = df.iloc[:, 1:5]

y = df.iloc[:, 5]
#importing some resources for preprocessing, creating pipeline, creating splits from dataset and CV

from sklearn.pipeline import make_pipeline

from sklearn import preprocessing

from sklearn.model_selection import  ShuffleSplit

from sklearn.model_selection import cross_val_score
#Applying SVM

from sklearn import svm



clf_svm = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = 1))

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

scores_svm = cross_val_score(clf_svm, X, y, cv = cv)



print("Accuracy of SVM: %0.3f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std()))
#Applying Knn

from sklearn.neighbors import KNeighborsClassifier



clf_knn = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski'))

scores_knn = cross_val_score(clf_knn, X, y, cv = cv)



print("Accuracy of Knn: %0.3f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std()))
#Applying XGBoost

import xgboost as xgb



clf_xgb = make_pipeline(preprocessing.StandardScaler(), xgb.XGBClassifier())

scores_xgb = cross_val_score(clf_xgb, X, y, cv = cv)



print("Accuracy of XGBoost: %0.3f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std()))
#Applying Decision Tree

from sklearn import tree



clf_tree = make_pipeline(preprocessing.StandardScaler(), tree.DecisionTreeClassifier(criterion='gini'))

scores_tree = cross_val_score(clf_tree, X, y, cv = cv)



print("Accuracy of Decision Tree: %0.3f (+/- %0.2f)" % (scores_tree.mean(), scores_tree.std()))
#Applying Random Forest

from sklearn.ensemble import RandomForestClassifier



clf_rf = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())

scores_rf = cross_val_score(clf_rf, X, y, cv = cv)



print("Accuracy of Random Forest: %0.3f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std()))