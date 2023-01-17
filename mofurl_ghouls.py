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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.describe()
train.info()
test.info()
train.drop("id", axis=1, inplace=True)
sns.pairplot(train, hue="type")
sns.countplot(train["color"], hue=train["type"])
train.drop("color", axis=1, inplace=True)
test.drop("color", axis=1, inplace=True)
X_train = train.drop("type", axis=1)
columns = X_train.columns
y_train = train["type"].copy()
X_test = test.drop("id", axis=1)
Id = test["id"].copy()
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
from sklearn.model_selection import train_test_split

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
def modeling(params, estimator):
    '''
    receive hyper paramaters and a model,
    (execute GridSearchCV using 5 folds
    print best hyper paramaters
    print accuracy about the model using the hyper paramater by testing validation data)
    return accuracy and the model
    '''
    
    grid = GridSearchCV(estimator, params, scoring="accuracy", n_jobs=-1)
    grid.fit(X_training, y_training)
    
    clf = grid.best_estimator_
    clf.fit(X_training, y_training)
    predict = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, predict)
    print("paramater:", grid.best_params_)
    print("accuracy:", accuracy)
    
    return accuracy, grid.best_estimator_
params = {"n_estimators": [5, 10, 20, 25],
         "max_depth": [3, 5, 7, 9, None],
         "max_features": ["auto", "sqrt", "log2", None]}

rfc_accuracy, rfc_clf = modeling(params, RandomForestClassifier())
importance = pd.DataFrame({"feature": columns, "importance": rfc_clf.feature_importances_})
importance.sort_values(by="importance", ascending=False)
params = {"C": [0.5, 1.0, 1.5],
         "gamma": [0.01, 0.05, 0.1],
         "probability": [True]}

svc_accuracy, svc_clf = modeling(params, SVC())
params =  {"C": [0.1, 1, 10],
          "max_iter": [50, 100, 200]}

lr_accuracy, lr_clf = modeling(params, LogisticRegression())
params = {"n_neighbors": [2, 3, 4, 5, 10, 15],
         "leaf_size": [20, 30, 50],
         "weights": ["uniform", "distance"],
         "algorithm": ["auto", "ball_tree", "kd_tree"]}

knc_accuracy, knc_clf = modeling(params, KNeighborsClassifier())
params = {}

gnb_accuracy, gnb_clf = modeling(params, GaussianNB())
params = {"C": [0.005, 0.01, 0.5, 1.0]}
    
lsvc_accuracy, lsvc_clf = modeling(params, LinearSVC())
params = {"learning_rate": [0.01, 0.03, 0.05, 0.1],
         "n_estimators": [30, 50, 100]}

gbc_accuracy, gbc_clf = modeling(params, GradientBoostingClassifier())
accuracy = pd.DataFrame({"model": ["RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "LinearSVC", "GradientBoostingClassifier"],
                        "accuracy": [rfc_accuracy, svc_accuracy, lr_accuracy, knc_accuracy, gnb_accuracy, lsvc_accuracy, gbc_accuracy]})
accuracy.sort_values(by="accuracy", ascending=False)
from sklearn.ensemble import VotingClassifier

vt_clf = VotingClassifier(estimators=[("lr", lr_clf), ("svc", svc_clf), ("gbc", gbc_clf)], voting="soft")
vt_clf.fit(X_training, y_training)
print("accuracy:", accuracy_score(vt_clf.predict(X_valid), y_valid))
lr_clf.fit(X_train, y_train)
submission_prediction = lr_clf.predict(X_test)
submission = pd.DataFrame({"id": Id, "type": submission_prediction})
submission.to_csv("submission.csv", index=False)