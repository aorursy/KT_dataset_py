# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
train_df.info()
train_df.describe().transpose()
sns.countplot(train_df['Survived'])
sns.countplot(train_df['Sex'])
def impute(passenger):
    age,sex=passenger
    if age < 18:
        return 'child'
    else:
        return sex
train_df['person']=train_df[['Age','Sex']].apply(impute,axis=1)
sns.countplot(train_df['person'],hue=train_df['Survived'])
sns.countplot(train_df['Pclass'],hue=train_df['person'])
sns.countplot(train_df['Pclass'],hue=train_df['Survived'])
train_df['Age'].hist(bins=70)
sns.boxplot(train_df['Age'])
fig = sns.FacetGrid(train_df, hue="Sex",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig = sns.FacetGrid(train_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig = sns.FacetGrid(train_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
deck=train_df['Cabin'].dropna()
levels = []

for level in deck:
    levels.append(level[0])    

cabin_df = pd.DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d')
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.countplot('Cabin',data=cabin_df,palette='summer')
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.countplot('Cabin',hue=train_df['Survived'],data=cabin_df,palette='summer')
sns.countplot('Embarked',data=train_df,hue='Pclass')
sns.countplot('Embarked',data=train_df,hue='Survived')
train_df.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
test_df.drop(['Ticket','Cabin'],axis=1,inplace=True)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 24
    else:
        return Age
train_df['Age']=train_df[['Age','Pclass']].apply(impute_age,axis=1)
test_df['Age']=test_df[['Age','Pclass']].apply(impute_age,axis=1)
train_df['Ageband']=pd.cut(train_df['Age'],5)
train_df[['Ageband','Survived']].groupby(['Ageband'],as_index=False).mean().sort_values(by='Ageband')
def impute(col):
    if col<=16:
        return 0
    if col>16 and col<=32:
        return 1
    if col>32 and col<=48:
        return 2
    if col>48 and col<=64:
        return 3
    if col>64:
        return 4
train_df['Age']=train_df['Age'].apply(impute)
test_df['Age']=test_df['Age'].apply(impute)
combine=[train_df]
for i in combine:
    i['Title']=i.Name.str.extract(' ([A-Za-z]+)\.',expand = True)
pd.crosstab(train_df['Title'],train_df['Sex'])
for i in combine:
    i['Title'] = i['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    i['Title'] = i['Title'].replace('Mlle', 'Miss')
    i['Title'] = i['Title'].replace('Ms', 'Miss')
    i['Title'] = i['Title'].replace('Mme', 'Mrs')
combine=[test_df]
for i in combine:
    i['Title']=i.Name.str.extract(' ([A-Za-z]+)\.',expand = True)
pd.crosstab(test_df['Title'],train_df['Sex'])
for i in combine:
    i['Title'] = i['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    i['Title'] = i['Title'].replace('Mlle', 'Miss')
    i['Title'] = i['Title'].replace('Ms', 'Miss')
    i['Title'] = i['Title'].replace('Mme', 'Mrs')
train_df['Fareband']=pd.qcut(train_df['Fare'],4)
train_df[['Fareband','Survived']].groupby(train_df['Fareband'],as_index=True).mean().sort_values(by='Fareband')
def impute(col):
    if col<=7.91:
        return 0
    if col>7.91 and col<=14.454:
        return 1
    if col>14.454 and col<=31.0:
        return 2
    if col>31.0:
        return 3
train_df['Fare']=train_df['Fare'].apply(impute)
test_df['Fare']=test_df['Fare'].apply(impute)
train_df['isalone']=train_df['Parch']+train_df['SibSp']
test_df['isalone']=test_df['Parch']+test_df['SibSp']
def impute(col):
    if col>0:
        return 1
    else:
        return 0
train_df['isalone']=train_df['isalone'].apply(impute)
test_df['isalone']=test_df['isalone'].apply(impute)
train_df.drop(['Name','person','Ageband','Fareband'],axis=1,inplace=True)
test_df.drop('Name',axis=1,inplace=True)
train_df['Title']=train_df['Title'].map({'Mr':1,'Mrs':2,'Miss':3,'Master':4,'Rare':5})
test_df['Title']=test_df['Title'].map({'Mr':1,'Mrs':2,'Miss':3,'Master':4,'Rare':5})
test_df['Fare'] = test_df['Fare'].fillna(0,inplace=True)
train_df=pd.get_dummies(train_df,drop_first=True)
test_df=pd.get_dummies(test_df,drop_first=True)
train_df.head()
X_train=train_df.drop('Survived',axis=1)
y_train=train_df['Survived']
X_test=test_df
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
prediction=log_reg.predict(X_test)
acc_reg=round(log_reg.score(X_train,y_train)*100,2)
print(acc_reg)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
prediction=knn.predict(X_test)
acc_knn=round(knn.score(X_train,y_train)*100,2)
print(acc_knn)
gaussian = GaussianNB()
gaussian.fit(X_train,y_train)
prediction=gaussian.predict(X_test)
acc_gaussian=round(gaussian.score(X_train,y_train)*100,2)
acc_gaussian
svc=SVC()
svc.fit(X_train,y_train)
prediction=svc.predict(X_test)
acc_svc=round(svc.score(X_train,y_train)*100,2)
acc_svc
dtr=DecisionTreeClassifier()
dtr.fit(X_train,y_train)
prediction=dtr.predict(X_test)
acc_dtr=round(dtr.score(X_train,y_train)*100,2)
print(acc_dtr)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
prediction=rfc.predict(X_test)
acc_rfc=round(rfc.score(X_train,y_train)*100,2)
print(acc_rfc)
models=pd.DataFrame({'Models':['LogisticRegression','KNeighborsClassifier','GaussianNB','SVC','DecisionTreeClassifier','RandomForestClassifier'],
        'Score':[acc_reg,acc_knn,acc_gaussian,acc_svc,acc_dtr,acc_rfc]})
models.sort_values(by='Score',ascending=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction})
submission.to_csv('my_submission.csv', index=False)
print('Submitted!')