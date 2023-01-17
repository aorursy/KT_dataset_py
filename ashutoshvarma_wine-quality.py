import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn import tree

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
red = pd.read_csv('../input/winequality-red.csv')
red
red.isnull().sum()
red.info()
sns.barplot(x='fixed acidity',y='quality',data=red)
sns.jointplot(x='alcohol',y='quality',data=red)
X = red.drop(columns='quality')
y = red['quality']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
X
y
standard = StandardScaler()

X_train_scaled = standard.fit(X_train)
print(X_train_scaled)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)
print('f1 : score',f1_score(y_pred1,y_test,average='weighted'))

print('accuracy =',accuracy_score(y_pred1,y_test))
print('classification report',classification_report(y_pred1,y_test))
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('f1 : score',f1_score(y_pred,y_test,average='weighted'))
print('accuracy =',accuracy_score(y_pred,y_test))

print('classification report',classification_report(y_pred,y_test))