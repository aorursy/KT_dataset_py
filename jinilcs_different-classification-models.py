# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv("../input/data.csv")
df.head(3)
X=df[df.columns[2:32]]

y=df[df.columns[1]]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, stratify=y)
X_train.shape
X_test.shape
knr = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)

print("train score - " + str(knr.score(X_train, y_train)))

print("test score - " + str(knr.score(X_test, y_test)))
lr1 = LogisticRegression(random_state=0).fit(X_train, y_train)

print("train score - " + str(lr1.score(X_train, y_train)))

print("test score - " + str(lr1.score(X_test, y_test)))
lr2 = LogisticRegression(C=6, random_state=0).fit(X_train, y_train)

print("train score - " + str(lr2.score(X_train, y_train)))

print("test score - " + str(lr2.score(X_test, y_test)))
svc = LinearSVC(random_state=0).fit(X_train,y_train)

print("train score - " + str(svc.score(X_train, y_train)))

print("test score - " + str(svc.score(X_test, y_test)))
svc = LinearSVC(C=3, random_state=0).fit(X_train,y_train)

print("train score - " + str(svc.score(X_train, y_train)))

print("test score - " + str(svc.score(X_test, y_test)))
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



svc_scaled = LinearSVC(C=2, random_state=0).fit(X_train_scaled,y_train)

print("train score - " + str(svc_scaled.score(X_train_scaled, y_train)))

print("test score - " + str(svc_scaled.score(X_test_scaled, y_test)))
dec = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

print("train score - " + str(dec.score(X_train, y_train)))

print("test score - " + str(dec.score(X_test, y_test)))
dec = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)

print("train score - " + str(dec.score(X_train, y_train)))

print("test score - " + str(dec.score(X_test, y_test)))
forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)

print("train score - " + str(forest.score(X_train, y_train)))

print("test score - " + str(forest.score(X_test, y_test)))
forest = RandomForestClassifier(n_estimators=100, max_features=30, max_depth=5, random_state=0).fit(X_train, y_train)

print("train score - " + str(forest.score(X_train, y_train)))

print("test score - " + str(forest.score(X_test, y_test)))
gb = GradientBoostingClassifier().fit(X_train, y_train)

print("train score - " + str(gb.score(X_train, y_train)))

print("test score - " + str(gb.score(X_test, y_test)))
gb = GradientBoostingClassifier(random_state=0, learning_rate=0.15).fit(X_train, y_train)

print("train score - " + str(gb.score(X_train, y_train)))

print("test score - " + str(gb.score(X_test, y_test)))
print("Support Vector Machine test score - " + str(svc_scaled.score(X_test_scaled, y_test)))

print("Gradient Boosting test score - " + str(gb.score(X_test, y_test)))

print("Logistic Regression test score - " + str(lr2.score(X_test, y_test)))