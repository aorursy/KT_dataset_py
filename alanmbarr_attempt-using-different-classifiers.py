import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import ensemble, cluster, neighbors, naive_bayes, svm, preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from subprocess import check_output

import xgboost as xgb



dftrain = pd.read_csv("../input/train.csv")

dftest = pd.read_csv("../input/test.csv")



y = dftrain['type']

dftrain = dftrain.drop(["type","id","color"],axis=1)



#dftrain = pd.get_dummies(dftrain)

X_train, X_test, y_train, y_test = train_test_split(dftrain, y, test_size=0.3, random_state=0)

rfc = ensemble.RandomForestClassifier(n_estimators=1000,max_depth=7,n_jobs=-1)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 

print(rfc.score(X_train,y_train))

print(rfc.score(X_test,y_test))

print(classification_report(y_pred,y_test))
clf = naive_bayes.GaussianNB()

clf.fit(X_train,y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))

y_pred = clf.predict(X_test)



print(classification_report(y_pred,y_test))
clf = svm.LinearSVC()

clf.fit(X_train,y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))



y_pred = clf.predict(X_test)



print(classification_report(y_pred,y_test))
clf = ensemble.AdaBoostClassifier()

clf.fit(X_train,y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))



y_pred = clf.predict(X_test)



print(classification_report(y_pred,y_test))
gbm = xgb.XGBClassifier(max_depth=7, n_estimators=100, learning_rate=0.05)

gbm.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))

print(classification_report(y_pred,y_test))