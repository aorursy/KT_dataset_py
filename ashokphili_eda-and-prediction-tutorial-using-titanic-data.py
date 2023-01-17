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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.shape

test.shape
print(train.columns)

print(test.columns)
train.dtypes
train.head()
train['Pclass'].value_counts()
train[['Survived','Pclass']].groupby(['Pclass']).mean().sort_values('Survived',ascending=False)
train[['Survived','Pclass']].groupby(['Pclass']).mean().sort_values('Survived',ascending=False).plot.bar()
train['Title']=train['Name'].str.extract('([A-Za-z]+)\.')
train['Title'].value_counts()
pd.crosstab(train['Title'], train['Sex'])
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)
train.head()
train.Age.isnull().sum()

train['Age']=train.Age.fillna(train.Age.mean())
sns.distplot(train['Age'])
train['Age-band']=pd.cut(train['Age'],5)
train['Age-band'].value_counts()
train.loc[train['Age']<16,'Age']=1

train.loc[(train['Age']>=16)&(train['Age']<32),'Age']=2

train.loc[(train['Age']>=32)&(train['Age']<48),'Age']=3

train.loc[(train['Age']>=48)&(train['Age']<64),'Age']=4

train.loc[(train['Age']>=64),'Age']  =5                           
train.Age.value_counts()
train['Fare-band']=pd.qcut(train['Fare'],4)
train['Fare-band'].value_counts()
train.loc[train['Fare']<7,'Fare']=1

train.loc[(train['Fare']>=7)&(train['Fare']<14),'Fare']=2

train.loc[(train['Fare']>=14)&(train['Fare']<31),'Fare']=3

train.loc[(train['Fare']>=31),'Fare']=4

train['Fare'].value_counts()
train[['Survived','Fare']].groupby('Fare').mean().sort_values('Survived',ascending=False)
train[['Survived','Fare']].groupby('Fare').mean().sort_values('Survived',ascending=False).plot.bar()
train['FamilySize']=train['SibSp']+train['Parch']+1

train['FamilySize'].value_counts()
train[['Survived','FamilySize']].groupby('FamilySize').mean().sort_values('Survived',ascending=False)
train['Cabin'].isnull().sum()
train.drop('Cabin',axis=1,inplace=True)
train['Embarked'].isnull().sum()
train['Embarked'].value_counts()
train.Embarked.fillna('S',inplace=True)
train[['Survived','Embarked']].groupby('Embarked').mean().sort_values('Survived',ascending=False)
train.head()
drop_columns=['Name','SibSp','Parch','Ticket','Age-band','Fare-band','PassengerId']

train.drop(drop_columns,axis=1,inplace=True)
train.head()

train['Sex']=train.Sex.map({'male':0,'female':1})

train['Embarked']=train.Embarked.map({'S':0,'C':1,'Q':2})

train.head()

cat_cols = ['Pclass', 'Age', 'Fare', 'Embarked', 'Title','FamilySize']

train= pd.get_dummies(train, columns = cat_cols,drop_first=True)

train.head()

X=train.iloc[:,1:].values

y=train.iloc[:,0].values

X.shape

y.shape
#split dataset into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler().fit(X_train)

#X_train = scaler.transform(X_train)

#X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),

          'Random Forest': RandomForestClassifier(n_estimators=10), 

          'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=1),

            'Linear SVM':SVC(kernel='rbf', gamma=.10, C=1.0)}

accuracy={}

for descr,model in models.items():

    mod=model

    mod.fit(X_train,y_train)

    prediction=mod.predict(X_test)    

    accuracy[descr]=((prediction==y_test).mean())

print(accuracy)


