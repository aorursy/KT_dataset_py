import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
dataset.head()
dataset.shape
dataset.info()
per=dataset['Class'][dataset['Class']==1].count()/(dataset['Class'][dataset['Class']==0].count())

per
Classes=['Not Fraud','Fraud']

fraud_fraudnot = pd.value_counts(dataset['Class'], sort = True)

fraud_fraudnot.plot(kind = 'bar', rot=0)

plt.title("Fraud not Fraud Distribution")

plt.xticks(range(2), Classes)

plt.xlabel("Class")

plt.ylabel("Frequency");
X=dataset.loc[:,:'Amount']

Y=dataset.loc[:,'Class']

X.shape,Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report,accuracy_score

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)



n_errors = (y_pred != y_test).sum()



print("Accuracy Score :")

print(accuracy_score(y_test,y_pred))

print("Classification Report :")

print(classification_report(y_test,y_pred))
from sklearn.linear_model import LogisticRegression

logregg=LogisticRegression()

y_pred=logregg.fit(X_train,y_train).predict(X_test)



n_errors = (y_pred != y_test).sum()



print("Accuracy Score :")

print(accuracy_score(y_test,y_pred))

print("Classification Report :")

print(classification_report(y_test,y_pred))
from sklearn import tree

clf = tree.DecisionTreeClassifier()

y_pred=clf.fit(X_train,y_train).predict(X_test)



n_errors = (y_pred != y_test).sum()



print("Accuracy Score :")

print(accuracy_score(y_test,y_pred))

print("Classification Report :")

print(classification_report(y_test,y_pred))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

y_pred=clf.fit(X_train,y_train).predict(X_test)



n_errors = (y_pred != y_test).sum()



print("Accuracy Score :")

print(accuracy_score(y_test,y_pred))

print("Classification Report :")

print(classification_report(y_test,y_pred))