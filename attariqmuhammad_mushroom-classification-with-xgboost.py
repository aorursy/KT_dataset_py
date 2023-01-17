import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
#load the dataset

df=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.head(10)
df.info()
y_cat = df['class'].values  

X_cat = df.drop('class', axis=1)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()



y = encoder.fit_transform(y_cat)

X = pd.get_dummies(X_cat).values
print(X)
print(y)
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 81)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()



gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)



cv_scores = cross_val_score(gnb, X, y, cv=10)



print(confusion_matrix(y_test, y_pred))

print("accuracy score : {}".format(accuracy_score(y_test,y_pred)))

print(classification_report(y_test, y_pred))



print(cv_scores)

print("average 10 fold tree : {}".format(np.mean(cv_scores)))



y_pred_proba=gnb.predict_proba(X_test)[:,1]

print("ROC AUC Score GNB : {}".format(roc_auc_score(y_test, y_pred_proba)))
from sklearn.tree import DecisionTreeClassifier



clf=DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



cv_scores = cross_val_score(clf, X, y, cv=10)



print(confusion_matrix(y_test, y_pred))

print("accuracy score : {}".format(accuracy_score(y_test,y_pred)))

print(classification_report(y_test, y_pred))



print(cv_scores)

print("average 10 fold tree : {}".format(np.mean(cv_scores)))



y_pred_proba=clf.predict_proba(X_test)[:,1]



print("ROC AUC Score CLF : {}".format(roc_auc_score(y_test, y_pred_proba)))
from xgboost import XGBClassifier

xgb = XGBClassifier()



xgb.fit(X_train, y_train)

xgb.fit(X_train, y_train)

y_pred = clf.predict(X_test)



cv_scores = cross_val_score(clf, X, y, cv=10)



print(confusion_matrix(y_test, y_pred))

print("accuracy score : {}".format(accuracy_score(y_test,y_pred)))

print(classification_report(y_test, y_pred))



print(cv_scores)

print("average 10 fold tree : {}".format(np.mean(cv_scores)))



y_pred_proba=xgb.predict_proba(X_test)[:,1]



print("ROC AUC Score xgb : {}".format(roc_auc_score(y_test, y_pred_proba)))