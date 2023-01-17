#IMPORTING LIBRARIES

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
# Proposed themes: darkgrid, whitegrid, dark, white, and ticks

#sns.set_style("whitegrid")

#sns.boxplot(data=data)

#plt.title("whitegrid")

def bar_chart(feature):

    survived=train[train['Survived']==1][feature].value_counts()

    dead=train[train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=['survived','dead']

    df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
train.head(3)
from scipy.misc import imread

from pylab import imshow, show



imshow(imread('titanic-pic.jpg'))

show()
alldata=[train,test] #combining the train and test dataset

for dataset in alldata:

    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)
train.Title.value_counts()
test.Title.value_counts()
title_mapping={'Mr':0,'Mrs':1,'Miss':2,'Master':3,

               'Dr':3,'Rev':3,'Col':3,'Major':3,'Mlle':3,'Countess':3,'Ms':3,'Lady':3,'Jonkheer':3,

               'Don':3,'Mme':3,'Capt':3,'Sir':3,'Dona':3,} 

for dataset in alldata:

    dataset['Title']=dataset['Title'].map(title_mapping)

                                                                                                                 
train.head(3)
test.head(3)
bar_chart('Title')
train.drop('Name',axis=1, inplace=True)

test.drop('Name', axis=1,inplace=True)
train.head(2)
test.head(2)
sex_mapping={'male':0,'female':1}

for dataset in alldata:

    dataset['Sex']=dataset['Sex'].map(sex_mapping)
train.head(2)
bar_chart('Sex')
train[train.Age.isnull()].shape
train.Age.fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

test.Age.fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(0,20)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(20,35)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(35,40)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(60)
for dataset in alldata:

    dataset.loc[ dataset['Age']<= 15, 'Age']=0,

    dataset.loc[(dataset['Age']>15) & (dataset['Age']<=35),'Age']=1,

    dataset.loc[(dataset['Age']>35) & (dataset['Age']<=45),'Age']=2,

    dataset.loc[(dataset['Age']>45) & (dataset['Age']<=60),'Age']=3,

    dataset.loc[ dataset['Age']>60, 'Age']=4
train.head()
test.head()
bar_chart('Age')
#checking the majority of embarkation from various classes

Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()

df=pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index=['1st class','2nd class','3rd class']

df.plot(kind='bar',stacked=True,figsize=(10,5))
for dataset in alldata:

    dataset['Embarked']=dataset['Embarked'].fillna('S')
train.isnull().sum().sort_values(ascending=False)
embarked_mapping={'S':0,'C':1,'Q':2}

for dataset in alldata:

    dataset["Embarked"]=dataset['Embarked'].map(embarked_mapping)
train.head()
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train.Fare.max()))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()

plt.xlim(0,20)
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()

plt.xlim(20,40)
for dataset in alldata:

    dataset.loc[ dataset['Fare']<= 15, 'Fare']=0,

    dataset.loc[(dataset['Fare']>15) & (dataset['Fare']<=30),'Fare']=1,

    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare']=2,

    dataset.loc[(dataset['Fare']>100),'Fare']=3
train.head()
train.Cabin.value_counts()
for dataset in alldata:

    dataset['Cabin']=dataset['Cabin'].str[:1]
#checking the majority of cabin from various classes

Pclass1=train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2=train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3=train[train['Pclass']==3]['Cabin'].value_counts()

df=pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index=['1st class','2nd class','3rd class']

df.plot(kind='bar',stacked=True,figsize=(10,5))
cabin_mapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8}

for dataset in alldata:

    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)

train.head()
#fill the missing cabin values with median cabin value of each Pclass

train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

test['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

train.head()
train['Family_size']=train['SibSp']+train['Parch']+1

test['Family_size']=test['SibSp']+test['Parch']+1

train.head()
facet=sns.FacetGrid(train,hue='Survived',aspect=4)

facet.map(sns.kdeplot,'Family_size',shade=True)

facet.set(xlim=(0,train['Family_size'].max()))

facet.add_legend()
#changing continous family size variable into categorical values, we use mapping

family_mapping={1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0}

for dataset in alldata:

    dataset['Family_size']=dataset['Family_size'].map(family_mapping)

train.head()
#dropping Ticket, SibSp and Parch columns

features_drop=['SibSp','Parch','Ticket']

train=train.drop(features_drop,axis=1)

test=test.drop(features_drop,axis=1)

train=train.drop(['PassengerId'],axis=1)

train.head(2)
test.head()
train_data=train.drop('Survived',axis=1)

target=train['Survived']
train_data.shape
target.shape
#importing classifier modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold=KFold(n_splits=10,shuffle=True,random_state=0)
clf=KNeighborsClassifier(n_neighbors=13)

scoring= 'accuracy'

score=cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=1,scoring=scoring)

print(score)

round(np.mean(score)*100,2)
clf=KNeighborsClassifier(n_neighbors=13)

clf.fit(train_data,target)

test_data=test.drop('PassengerId',axis=1).copy()

prediction=clf.predict(test_data)
submission=pd.DataFrame({

    'PassengerId':test['PassengerId'],

    'Survived':prediction

})

submission.to_csv('submission.csv',index=False)
submission=pd.read_csv('gender_submission.csv')
submission.head()