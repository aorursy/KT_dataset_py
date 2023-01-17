# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/StudentsPerformance.csv")
data.shape
data.columns
data[:15]
data["passes math"] = data["math score"] >= 60
data["passes reading"] = data["reading score"] >= 60
data["passes writing"] = data["writing score"] >= 60
data[:15]
data["gender"].value_counts()
data["race/ethnicity"].value_counts()
data["parental level of education"].value_counts()
data["lunch"].value_counts()
data["test preparation course"].value_counts()
sns.relplot(x="reading score", y="writing score", data=data)
sns.relplot(x="reading score", y="math score", data=data)
sns.relplot(x="writing score", y="math score", data=data)
sns.relplot(x="reading score", y="math score", hue="gender", data=data)
sns.relplot(x="reading score", y="math score", hue="lunch", data=data)
sns.relplot(x="reading score", y="math score",
            hue="test preparation course", data=data)
sns.catplot(x="parental level of education", y="math score",
            kind="swarm", data=data)
sns.catplot(x="parental level of education", y="math score",
            hue="gender", kind="swarm", data=data)
sns.catplot(x="parental level of education", y="math score",
            hue="gender", kind="box", data=data)
sns.catplot(x="gender", y="passes math", hue="lunch", kind="bar", data=data)
sns.catplot(x="parental level of education", y="passes math",
            hue="lunch", kind="bar", data=data)
sns.catplot(x="passes math", y="passes reading",
            hue="lunch", kind="bar", data=data)
sns.catplot(x="passes math", y="passes reading",
            hue="gender", kind="bar", data=data)
data.dtypes
data[data.isnull().any(axis=1)]
categorical_features = ["gender", "race/ethnicity", "parental level of education",
                       "lunch", "test preparation course"]
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features)
])
X = data.drop("math score", axis=1)
X = data.drop("reading score", axis=1)
X = data.drop("writing score", axis=1)
X = data.drop("passes reading", axis=1)
X = data.drop("passes writing", axis=1)
y = data["passes math"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def f(classifier, title):
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", classifier)])
    clf.fit(X_train, y_train)
    print("*" * 80)
    print("Score for classifier:", title)
    print(clf.score(X_test, y_test))
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
    cf = confusion_matrix(y_train, y_train_pred)
    print("Confusion matrix for classifier:", title)
    print(cf)
    print("Precision score for classifier:", title)
    print(precision_score(y_train, y_train_pred))
    print("Recall score for classifier:", title)
    print(recall_score(y_train, y_train_pred))
    print("*" * 80)
f(LogisticRegression(solver="lbfgs"), "Logistic Regression with lbfgs solver")
f(LogisticRegression(solver="liblinear"), "Logistic Regression with linear solver")
f(LogisticRegression(solver="newton-cg"), "Logistic Regression with newton-cg solver")
f(LogisticRegression(solver="sag"), "Logistic Regression with sag solver")
f(SGDClassifier(random_state=42), "SGD Classifier")
f(KNeighborsClassifier(), "K Neighbors Classifier")
f(SVC(), "Support Vector Machine Classifier")
f(RandomForestClassifier(n_estimators=10), "Random Forest Classifier, n=10")
f(DecisionTreeClassifier(), "Decision Tree Classifier")
clf_final = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", LogisticRegression(solver="lbfgs"))])
clf_final.fit(X_test, y_test)
print(clf_final.score(X_test, y_test))
y_test_pred = cross_val_predict(clf_final, X_test, y_test, cv=3)
cf_final = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:")
print(cf_final)
print("Precision score:", precision_score(y_test, y_test_pred))
print("Recall score:", recall_score(y_test, y_test_pred))
