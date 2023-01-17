

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

iris=load_iris()
#features

X=iris.data

X
iris.feature_names
#label

y=iris.target

y
#flower species

iris.target_names
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2) # train/test ratio is 80/20
from sklearn.neighbors import KNeighborsClassifier

kn=KNeighborsClassifier()

kn.fit(X_train,y_train)

kn.predict(X_test)

kn_score=kn.score(X_test,y_test)

kn_score
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=1)

lr.fit(X_train,y_train)

lr.predict(X_test)

lr_score=lr.score(X_test,y_test)

lr_score
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(random_state=1)

dt.fit(X_train,y_train)

dt.predict(X_test)

dt_score=dt.score(X_test,y_test)

dt_score
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=1)

rf.fit(X_train,y_train)

rf.predict(X_test)

rf_score=dt.score(X_test,y_test)

rf_score

from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()

gb.fit(X_train,y_train)

gb.predict(X_test)

gb_score=dt.score(X_test,y_test)

gb_score
print("KNN_score: ",kn_score)

print("logistic regression_score: ",lr_score)

print("decision tree_score: ",dt_score)

print("random forest_score: ",rf_score)

print("gradient boosting_score: ",gb_score)