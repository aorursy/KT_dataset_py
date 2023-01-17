# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# import models
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score as cv
train.head(10)
train.info()
test.info()
train.describe()
train.describe(include=['O'])
train.groupby(['Pclass'],as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
train.groupby(['Sex'],as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
train.groupby(['Parch'],as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
train.groupby(['SibSp'],as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
grph = sns.FacetGrid(train,col='Survived')
grph.map(plt.hist,'Age',alpha=0.5,bins=20)
grph=sns.FacetGrid(train,col='Survived',row='Pclass')
grph.map(plt.hist,'Age',bins=20,alpha=0.5)
grph = sns.FacetGrid(train,row='Embarked',size=2.2,aspect=1.6)
grph.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
grph.add_legend()
s=train.groupby(['Sex','Survived']).Survived.count()
m_count=train.groupby(['Sex']).Survived.count()[1]
f_count=train.groupby(['Sex']).Survived.count()[0]

f=s.loc[('female')]/f_count
m=s.loc[('male')]/m_count
f
width=0.15
plt.figure(figsize=(7,7))
plt.title('Survived segmented rate by Gender')

plt.bar(f.index,f,width,label=train.Sex.unique()[1])
plt.bar(m.index+width,m,width,label=train.Sex.unique()[0])
plt.xticks(m.index+width/2,[0,1])
plt.xlabel('Survived')
plt.ylabel('rate')
plt.legend(loc='best')
plt.show()
grph=sns.FacetGrid(train,row='Embarked',col='Survived',size=2.6,aspect=1.6)
grph.map(sns.barplot,'Sex','Fare',alpha=0.8)
grph.add_legend()
dataset=[train,test]
for data in dataset:
    data['Title']=data.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])
#train
for data in dataset:
    data['Title']=data['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Rare')
    data['Title']=data['Title'].replace(['Mlle','Ms'],'Miss')
    data['Title']=data['Title'].replace('Mme','Mrs')
train.groupby(['Title'],as_index=False).Survived.mean()
title={'Mr':1,'Mrs':2,'Miss':3,'Master':4,'Rare':5}
for data in dataset:
    data['Title']=data['Title'].map(title)
    data['Title']=data['Title'].fillna(0)
train
train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
dataset=[train,test]
train.shape,test.shape
for data in dataset:
    data['Sex']=data['Sex'].map({'male':0,'female':1}).astype(int)
data
age=np.zeros((2,3))
age
for data in dataset:
    for i in range(0,2):
        for j in range(0,3):
            age_drop=data[(data['Sex']==i)&(data['Pclass']==j+1)].Age.dropna()
            age_median=age_drop.median()
            age[i,j] = int( age_median/0.5 + 0.5 ) * 0.5
for data in dataset:
    for i in range(0,2):
        for j in range(0,3):
            data.loc[(data.Age.isnull())&(data.Sex==i)&(data.Pclass==j+1),'Age']=age[i,j]
            
train
train['AgeBand']=pd.cut(train['Age'],5)
train.groupby('AgeBand',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
for data in dataset:
    data.loc[data['Age']<=16,'Age']=0
    data.loc[(data['Age']>16) & (data['Age']<=32),'Age']=1
    data.loc[(data['Age']>32) & (data['Age']<=48),'Age']=2
    data.loc[(data['Age']>48) & (data['Age']<=64),'Age']=3
    data.loc[(data['Age']>64),'Age']=4
    data['Age']=data['Age'].astype(int)
train
train=train.drop(['AgeBand'],axis=1)
#test=test.drop(['AgeBand'],axis=1)
dataset=[train,test]
train
for data in dataset:
    data['Company']=data['SibSp']+data['Parch']+1
train.groupby('Company',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
for data in dataset:
    data['IsAlone']=0
    data.loc[data['Company']==1,'IsAlone']=1
train.groupby('IsAlone',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
train=train.drop(['SibSp','Parch','Company'],axis=1)
test=test.drop(['SibSp','Parch','Company'],axis=1)
dataset=[train,test]
print(test.shape)
train.shape
freq=train.Embarked.dropna().mode()[0]
freq
for data in dataset:
    data['Embarked']=data['Embarked'].fillna(freq)
train.groupby('Embarked',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
for data in dataset:
    data['Embarked']=data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train.groupby('Embarked',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()
train['FareBand']=pd.qcut(train['Fare'],4)
train.groupby('FareBand',as_index=False).Survived.mean().sort_values(by='Survived',ascending=False)
for data in dataset:
    data.loc[data['Fare']<=7.91,'Fare']=0
    data.loc[(data['Fare']>7.91) & (data['Fare']<=14.45),'Fare']=1
    data.loc[(data['Fare']>14.45) & (data['Fare']<=31),'Fare']=2
    data.loc[data['Fare']>31,'Fare']=3
    data['Fare']=data['Fare'].astype(int)
train.groupby('Fare',as_index=False).Survived.mean().sort_values(by='Fare',ascending=False)
train=train.drop('FareBand',axis=1)
print(train.shape)
test.shape
train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
train.head()
train['Age*Class'] = train.Age * train.Pclass
test['Age*Class'] = test.Age * test.Pclass
train
X_train=train.drop('Survived',axis=1)
y_train=train['Survived']
X_test=test.drop('PassengerId',axis=1)
logreg=LogisticRegression()
acc_log=cv(logreg,X_train,y_train,cv=5,scoring='accuracy').mean()
logreg.fit(X_train, y_train)
acc_log
coeff=pd.DataFrame(train.columns.delete(0))
coeff.columns=['Feature']
coeff['Correlation']=(logreg.coef_[0])
coeff.sort_values(by='Correlation',ascending=False)
svc = SVC(kernel='poly')
acc_svc = cv(svc,X_train, y_train,cv=5,scoring='accuracy').mean()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc
gbc=GradientBoostingClassifier(n_estimators=300)
acc_gbc=cv(gbc,X_train,y_train,cv=5,scoring='accuracy').mean()
gbc.fit(X_train,y_train)
Y_pred_Best=gbc.predict(X_test)
acc_gbc
rfc=RandomForestClassifier(n_estimators=100)
acc_rfc=cv(rfc,X_train,y_train,cv=5,scoring='accuracy').mean()
rfc.fit(X_train,y_train)
Y_pred=rfc.predict(X_test)
acc_rfc
knn = KNeighborsClassifier(n_neighbors = 3)
acc_knn=cv(knn,X_train,y_train,cv=5,scoring='accuracy').mean()
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn
linear_svc = LinearSVC()
acc_linear_svc = cv(linear_svc,X_train, y_train,cv=5,scoring='accuracy').mean()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc

sgd = SGDClassifier()
acc_sgd = cv(sgd,X_train, y_train,cv=5,scoring='accuracy').mean()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd
DT = DecisionTreeClassifier()
acc_DT = cv(DT,X_train, y_train,cv=5,scoring='accuracy').mean()
DT.fit(X_train, y_train)
Y_pred = DT.predict(X_test)
acc_DT
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Gradient Boosting',
              'Random Forest','Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_gbc,
              acc_rfc, acc_sgd, acc_linear_svc, acc_DT]})
models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_Best
    })
submission.to_csv('titanic_data.csv',index=False)
print('submission successful')
