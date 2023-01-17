# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

sns.set()

%matplotlib inline
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head(5)
train.shape
train.info()
train.describe()
train['Pclass'].value_counts()
train['SibSp'].value_counts()
train.Age.plot(kind='hist')

plt.xlabel("Age in years")

plt.show()

class_wise_survival=train[['Pclass','Survived']].groupby('Pclass').mean()

class_wise_survival
class_wise_survival.plot(kind='bar')
sex_wise_survival=train[['Sex','Survived']].groupby('Sex').mean()

sex_wise_survival
sex_wise_survival.plot(kind='bar')
Rel_wise_survival=train[['SibSp','Survived']].groupby('SibSp').mean()

Rel_wise_survival.sort_values(by='Survived',ascending=False)
train[['Parch','Survived']].groupby('Parch').mean().sort_values(by='Survived',ascending=False)
s=sns.FacetGrid(train,col='Survived',hue='Survived')

s.map(plt.hist,'Age')

plt.show()

s=sns.FacetGrid(train,col='Survived',hue='Survived')

s.map(plt.hist,'Pclass')

plt.show()
plotti
class_Sex_Survival=pd.pivot_table(train,index=['Sex'],columns='Pclass',values='Survived',aggfunc='mean')

class_Sex_Survival
sns.factorplot(x="Sex", y="Survived", col="Pclass",data=train, saturation=.5,

                 kind="bar", ci=None, aspect=.6)
class_Sex_Survival.boxplot()

plt.xlabel("Pclass")

plt.ylabel("Survival rate")

plt.show()
embark_sex_survival=pd.pivot_table(train,index='Sex',columns='Embarked',values='Survived',aggfunc='mean')

embark_sex_survival
sns.factorplot(x="Sex", y="Survived", col="Embarked",data=train, saturation=.5,

                 kind="bar", ci=None, aspect=.6)
embark_sex_survival.boxplot()

plt.xlabel("Embarked Region")

plt.ylabel("Survival rate")

plt.show()
plt.subplot(121)

sns.countplot(train["Embarked"], hue = train["Survived"])

plt.subplot(122)

sns.countplot(train['Pclass'], hue=train['Survived'])

plt.show()
s=sns.FacetGrid(train,col='Survived',row='Pclass',hue='Survived')

s.map(plt.hist,'Age')

plt.show()
#import cv2

#img = cv2.imread('titanic_layout.jpg',0)

#cv2.imshow('image',img)
train=train.drop(['Ticket','Cabin','PassengerId'],axis=1)

test=test.drop(['Ticket','Cabin','PassengerId'],axis=1)

combine=[train,test]



train.Age=train.Age.fillna(train.Age.median())

test.Age=test.Age.fillna(test.Age.median())

print(train.Age.count())

train.columns
for x in combine:

    x['Sex']=x['Sex'].map({'female':1,'male':0}).astype(int)

    #x['Embarked']=x['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)

train.head()
for i in combine:

    i['age_range']=pd.cut(i['Age'],5)

    #i[['age_range','Survived']].groupby('age_range').mean()

#train['Age'].astype(int)
for x in combine:

    x.loc[(x.Age<=16),'Age']=0

    x.loc[(x.Age>16) & (x.Age<=32),'Age']=1

    x.loc[(x.Age>32) & (x.Age<=48),'Age']=2

    x.loc[(x.Age>48) & (x.Age<=64),'Age']=3

    x.loc[(x.Age)>64,'Age']=4

    x.Age=x.Age.astype(int)

   
#x.Age=x.Age.fillna(2)

for i in combine:

    i['FamilySize']=i['SibSp']+i['Parch']+1

train[['FamilySize','Survived']].groupby("FamilySize").mean()
for i in combine:

    i['IsAlone']=0

    i.loc[(i.IsAlone==1),'IsAlone']=1
train=train.drop(['FamilySize','Parch','SibSp','age_range'],axis=1)

test=test.drop(['FamilySize','Parch','SibSp','age_range'],axis=1)

combine=[train,test]
train.head(5)
combine=[train,test]

for i in combine:

    freq=i.Embarked.mode()

    i.Embarked=i.Embarked.fillna(freq[0])

    i['Embarked']=i.Embarked.map({'S':0,'C':1,'Q':2}).astype(int)

train.info()
train[['Embarked','Survived']].groupby(["Embarked"], as_index=False).mean()



train.head()
train.Embarked.value_counts()

test.Embarked.value_counts()
test['Fare']=test['Fare'].fillna(test.Fare.mean())
test.head()
for i in combine:

    i['Fare']=i['Fare'].round(2)

    i.loc[(i.Fare<=7.91),'Fare']=0

    i.loc[(i.Fare>7.91) & (i.Fare<=14.45),'Fare']=1

    i.loc[(i.Fare>14.4) & (i.Fare<=31.0),'Fare']=2

    i.loc[(i.Fare>31.0) & (i.Fare<=512.3),'Fare']=3

    i.Fare=i.Fare.astype(int)

train=train.drop(['Name'],axis=1)

test=test.drop(['Name'],axis=1)

train.head()
test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
X_train=train.drop(['Survived'],axis=1)

y_train=train['Survived']

X_test=test

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

logreg.score(X_train,y_train)
coeff=pd.DataFrame(train.columns.delete(0))

coeff.columns=['Features']

coeff['correlation']=pd.Series(logreg.coef_[0])

coeff.sort_values(by='correlation',ascending=False)
sv=SVC()

sv.fit(X_train,y_train)

y_pred=sv.predict(X_test)

sv.score(X_train,y_train)
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

knn.score(X_train,y_train)
gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

y_pred=gaussian.predict(X_test)

gaussian.score(X_train,y_train)
linear_svc=LinearSVC()

linear_svc.fit(X_train,y_train)

y_pred=linear_svc.predict(X_test)

linear_svc.score(X_train,y_train)
sgd=SGDClassifier()

sgd.fit(X_train,y_train)

y_pred=sgd.predict(X_test)

sgd.score(X_train,y_train)
decision_tree=DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)

y_pred=decision_tree.predict(X_test)

decision_tree.score(X_train,y_train)
random_forest=RandomForestClassifier()

random_forest.fit(X_train,y_train)

y_pred=random_forest.predict(X_test)

random_forest.score(X_train,y_train)
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

X=train.drop(['Survived'],axis=1)

y=train['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=23,test_size=0.3)
parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }
random_forest=RandomForestClassifier()

grid_obj=GridSearchCV(random_forest,parameters,scoring='accuracy')

grid_obj.fit(X_train,y_train)



est=grid_obj.best_estimator_

est_obj=est.fit(X_train,y_train)

y_pred=est_obj.predict(X_test)

est_obj.score(X_train,y_train)
print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))

print(classification_report(y_pred,y_test))