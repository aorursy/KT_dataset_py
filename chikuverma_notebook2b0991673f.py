import pandas as pd

import numpy as nm

from sklearn.neighbors import KNeighborsClassifier as knc

from sklearn.metrics import accuracy_score

from sklearn.cross_validation import train_test_split as tts, cross_val_score as cvs

from sklearn.linear_model import LogisticRegression as lr

import matplotlib.pyplot as plt
ti= pd.read_csv('F:/Study/train.csv',header=0,index_col=None)
ti.head()
print(ti.shape)

print(ti.dtypes)
ti.count()
ti= ti.drop(['Cabin','Name','Ticket','Embarked'], axis=1)
ti.head()
ti.Survived.mean()
Sex_Pclass=ti.groupby(['Pclass','Sex']).mean()

Sex_Pclass
%matplotlib inline

Sex_Pclass['Survived'].plot.bar()

Age_gp=pd.cut(ti["Age"], nm.arange(0, 90, 10))

Age_gpbar=ti.groupby(Age_gp).mean()

print(Age_gpbar)

Age_gpbar['Survived'].plot.bar()
ti.groupby(['Pclass','Sex']).mean()
ti.count()
ti['Age'].fillna(ti['Age'].median(), inplace=True)
ti.count()
ti.groupby(['Pclass','Sex']).mean()

ti.tail()
ti.groupby(['Pclass']).mean()

ti.groupby(['Pclass','Parch','Sex']).mean()
ti.describe()
print(ti.Age[29])

ti.count()
#analyzing the data graphically

survived_sex = ti[ti['Survived']==1]['Sex'].value_counts()

dead_sex = ti[ti['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(10,8))
figure = plt.figure(figsize=(10,8))

plt.hist([ti[ti['Survived']==1]['Age'], ti[ti['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
ax = plt.subplot()

ax.set_ylabel('Average fare')

ti.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(11,8), ax = ax)
ti=ti.replace('female', 0)

ti=ti.replace('male', 1)
ti.head()
ti.tail()
X = ti.drop(['Survived'], axis=1).values

y = ti['Survived'].values

print(X.shape)

y.shape
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
clf_lr = lr()
clf_lr.fit (X_train, y_train)

clf_lr.score (X_test, y_test)
from sklearn import tree

clf_dt = tree.DecisionTreeClassifier(max_depth=8,random_state=1)

clf_dt.fit (X_train, y_train)

clf_dt.score (X_test, y_test)
from sklearn.ensemble import RandomForestClassifier as rfc
clf_rf = rfc(n_estimators=51,random_state=1)

clf_rf.fit (X_train, y_train)

clf_rf.score (X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier as gbc
clf_gbc = gbc(n_estimators=54,random_state=1)

clf_gbc.fit (X_train, y_train)

clf_gbc.score (X_test, y_test)
from sklearn.svm import SVC as svc
clf_svc = svc(gamma=10, C=0.01)

clf_lr.fit (X_train, y_train)

clf_lr.score (X_test, y_test)

knn=knc(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
from sklearn.cross_validation import cross_val_score as cvs
print(cvs(clf_lr,X,y,cv=10,scoring="accuracy").mean())

print(cvs(clf_dt,X,y,cv=10,scoring="accuracy").mean())

print(cvs(clf_rf,X,y,cv=10,scoring="accuracy").mean())

print(cvs(clf_gbc,X,y,cv=10,scoring="accuracy").mean())

print(cvs(clf_svc,X,y,cv=10,scoring="accuracy").mean())

print(cvs(knn,X,y,cv=10,scoring="accuracy").mean())

tt= pd.read_csv('F:/Study/test.csv',header=0,index_col=None)
from sklearn.feature_selection import SelectFromModel

tt.head()
tt.count()
tt['Age'].fillna(tt['Age'].median(), inplace=True)
tt= tt.drop(['Cabin','Name','Ticket','Embarked'], axis=1)
tt=tt.replace('female', 0)

tt=tt.replace('male', 1)
tt.shape
tt.count()
tt['Fare'].fillna(tt['Fare'].median(), inplace=True)
tt.count()
output = clf_rf.predict(tt).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('F:/Study/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('F:/Study/output.csv',index=False)