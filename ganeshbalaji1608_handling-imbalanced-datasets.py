import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings

warnings. simplefilter(action = "ignore", category = Warning)
data = pd.read_csv("../input/creditcard/creditcard.csv")
data.head(5)
data['Class'].value_counts()
df = data.copy()
X = df.drop(columns = ['Class'])
Y = df[['Class']]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.7)
Y_train
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
classifier = RandomForestClassifier()

classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test, Y_pred))
from imblearn.under_sampling import NearMiss
ns = NearMiss(0.7)
X_train_ns, Y_train_ns = ns.fit_sample(X_train,Y_train)
print(Y_train['Class'].value_counts())
print(Y_train_ns['Class'].value_counts())
classifier = RandomForestClassifier()
classifier.fit(X_train_ns,Y_train_ns)
X_test_ns, Y_test_ns = ns.fit_sample(X_test,Y_test)
Y_pred = classifier.predict(X_test_ns)
print(confusion_matrix(Y_test_ns, Y_pred))
print(accuracy_score(Y_test_ns,Y_pred))
print(classification_report(Y_test_ns,Y_pred))
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler()
X_train_os, Y_train_os = os.fit_sample(X_train,Y_train)
print(Y_train['Class'].value_counts())
print(Y_train_os['Class'].value_counts())
classifier.fit(X_train_os,Y_train_os)
y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
#By giving sampling strategy of 0.5
os = RandomOverSampler(0.5)
X_train_os, Y_train_os = os.fit_sample(X_train, Y_train)
classifier.fit(X_train_os,Y_train_os)
y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
X_test_os,Y_test_os = os.fit_sample(X_test, Y_test)
y_pred = classifier.predict(X_test_os)
print(confusion_matrix(Y_test_os,y_pred))
print(accuracy_score(Y_test_os,y_pred))
print(classification_report(Y_test_os,y_pred))
from imblearn.combine import SMOTETomek
sm = SMOTETomek() #first try without giving the sampling strategy
X_train_sm, Y_train_sm = sm.fit_sample(X_train,Y_train)
print(Y_train['Class'].value_counts())
print(Y_train_sm['Class'].value_counts())
classifier = RandomForestClassifier()
classifier.fit(X_train_sm,Y_train_sm)
y_pred = classifier.predict(X_test)

print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
#trying with giving the sampling strategy

sm = SMOTETomek(0.5)
X_train_sm,Y_train_sm = sm.fit_sample(X_train,Y_train)
print(Y_train['Class'].value_counts())
print(Y_train_sm['Class'].value_counts())
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train_sm,Y_train_sm)
y_pred = classifier.predict(X_test)

print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
from imblearn.ensemble import EasyEnsembleClassifier
easy = EasyEnsembleClassifier()
easy.fit(X_train,Y_train)
y_pred = easy.predict(X_test)

print(confusion_matrix(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))