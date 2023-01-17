import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv', header=0)

data.head(5)
X = data.drop(columns=['species'], axis=1)

y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)
model.score(X_test, y_test)
model.score(X_train, y_train)
modelRF = RandomForestClassifier(n_estimators=10)

modelRF.fit(X_train,y_train)
modelRF.score(X_test, y_test)
modelRF.score(X_train, y_train)
model_bagging = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5, max_features=1, n_estimators=20)
model_bagging.fit(X_train, y_train)
model_bagging.score(X_test, y_test)
model_bagging = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=1)
model_bagging.fit(X_train, y_train)
model_bagging.score(X_test, y_test)
lr = LogisticRegression()

svm = SVC(kernel='poly', degree=2)

dt = DecisionTreeClassifier()
final_model = VotingClassifier(estimators=[('lr', lr),('dt', dt), ('svm', svm)], voting='hard')
final_model.fit(X_train, y_train)
final_model.score(X_test, y_test)