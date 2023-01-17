# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

data.head()
import seaborn as sns

sns.set(style="darkgrid")

sns.catplot(x='Survived',y='Age',hue='Sex',data=data)
sns.catplot(x='Survived',y='Age',hue='Sex',data=data,kind='point')
sns.catplot(x='Sex',y='Survived',hue='Pclass',data=data,kind='point')
sns.lmplot(x='Survived',y='Fare',col='Sex',row='Pclass',data=data,x_jitter=0.1)
data = data.replace({'male':1,'female':0,'S':0,'C':1,'Q':2})

test = test.replace({'male':1,'female':0,'S':0,'C':1,'Q':2})

data.head()
data.isna().sum()
data = data.drop('Cabin',axis=1)

test = test.drop('Cabin',axis=1)

data.head()
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'].mode()
data['Embarked'] = data['Embarked'].fillna(0)
test.isna().sum()
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
data.dtypes
test.dtypes
data['Age'] = data['Age'].astype(int)

data['Embarked'] = data['Embarked'].astype(int)
data.head()
test.head()
test['Age'] = test['Age'].astype(int)
data.groupby(['Sex','Survived']).size()
data.corr(method='pearson')
data.head()
data = data.drop(['Name','Ticket','PassengerId'],axis=1)

test = test.drop(['Name','Ticket','PassengerId'],axis=1)
data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

X = data.drop('Survived',axis=1)

Y = data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=10)

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)

print(classification_report(Y_test,Y_pred))

print(accuracy_score(Y_test,Y_pred))
#accuracy using CV

accuracy = cross_val_score(logreg,X,Y,cv=10)

print(accuracy.mean())
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,Y_train)

Y_predGNB = gnb.predict(X_test)

print(classification_report(Y_test,Y_predGNB))

print(accuracy_score(Y_test,Y_predGNB))
#Accuracy using CV 

accuracy_gnb = cross_val_score(gnb,X,Y,cv=10,scoring='accuracy')

print(accuracy_gnb.mean())
from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_SVC = svc.predict(X_test)

print(classification_report(Y_test,Y_pred_SVC))

print(accuracy_score(Y_test,Y_pred_SVC))
#Now using CV

accuracy_svc = cross_val_score(svc,X,Y,cv=10,scoring='accuracy')

print(accuracy_svc.mean())
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train,Y_train)

Y_pred_RF = clf.predict(X_test)

print(classification_report(Y_test,Y_pred_RF))

print(accuracy_score(Y_test,Y_pred_RF))            
#Now using CV

accuracy_rf = cross_val_score(clf,X,Y,cv=5,scoring='accuracy')

print(accuracy_rf.mean())