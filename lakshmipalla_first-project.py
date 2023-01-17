# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.chdir("../input/")





# Any results you write to the current directory are saved as output.
#Data Manipulation

import pandas as pd

import numpy as np



#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)

print(df_test.shape)
print(df_train.columns)

print(df_test.columns)
def find_feature(col):

    return df_train[col].unique()



list_columns =['Pclass','Sex','SibSp','Parch','Survived','Embarked']



for item in list_columns:

    print(item,find_feature(item))
print(df_train.info())

print(df_test.info())
print(df_train.describe())

print(df_test.describe())
print(df_train.dtypes)

print(df_test.dtypes)
sns.heatmap(df_train.isnull())
sns.heatmap(df_test.isnull())
print(df_train.isnull().sum())

print(df_test.isnull().sum())
def graph(feature):

    sns.countplot(x=feature)

    

graph(df_train['Survived'])    
graph(df_train['Pclass'])
graph(df_train['Sex'])
graph(df_train['SibSp'])
graph(df_train['Parch'])
graph(df_train['Embarked'])
df_train['Age'].plot.hist(x='Age')
df_train['Fare'].plot.hist(x='Fare')
#Variable : Survived

sns.countplot(x=df_train['Survived'])

print(df_train.groupby('Survived')['Survived'].count())

print('Survival Rate : ',df_train['Survived'].sum()/df_train['Survived'].count()*100,'%')
#Variable 'Pclass'

sns.countplot(x='Pclass',hue='Survived',data=df_train)

print(pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True))

print('*'*40)

print(df_train[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))
#Variable : Sex

sns.countplot(x='Sex',hue='Survived',data=df_train)

print(pd.crosstab(df_train['Sex'],df_train['Survived'],margins=True))

print('*'*40)

print(df_train[['Sex','Survived']].groupby('Sex',as_index=False).mean())
#Variable : Sibsp

sns.countplot(x='SibSp',hue='Survived',data=df_train)

print(pd.crosstab(df_train['SibSp'],df_train['Survived'],margins=True))

print('*'*40)

print(df_train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean())
#Variable : Parch

sns.countplot(x='Parch',data=df_train,hue='Survived')

print(df_train[['Parch','Survived']].groupby('Parch',as_index=False).mean())

print('*'*40)

print(pd.crosstab(df_train['Parch'],df_train['Survived'],margins=True))
#Varibale : Embarked

sns.countplot(x='Embarked',data=df_train,hue='Survived')

print(pd.crosstab(df_train['Embarked'],df_train['Survived'],margins=True))

print('*'*40)

print(df_train[['Embarked','Survived']].groupby('Embarked').mean())
df_train['Age'].fillna(df_train['Age'].median(),inplace=True)

df_test['Age'].fillna(df_test['Age'].median(),inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace=True)

df_test['Embarked'].fillna(df_test['Embarked'].mode()[0],inplace=True)
df_train.isnull().sum()
combine=[df_train,df_test]



for dataset in combine:

    dataset.loc[dataset["Age"] <= 16, "Age_group"] = 0

    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <=32), "Age_group"] = 1

    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <=48), "Age_group"] = 2

    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <=64), "Age_group"] = 3

    dataset.loc[(dataset["Age"] > 64), "Age_group"] = 4



df_train.head()    
#Variable 'Pclass'

sns.countplot(x='Age_group',hue='Survived',data=df_train)

print(pd.crosstab(df_train['Age_group'],df_train['Survived'],margins=True))

print('*'*40)

print(df_train[["Age_group", "Survived"]].groupby(['Age_group'],as_index=False).mean().sort_values(by='Survived',ascending=False))
for dataset in combine:

    dataset["Sex"].replace(["male", "female"], [0, 1], inplace = True)

    dataset["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace = True)
drop_columns_train=['PassengerId','Name','Age','Ticket','Fare','Cabin']

drop_columns_test=['Name','Age','Ticket','Fare','Cabin']

df_train.drop(drop_columns_train,axis=1,inplace=True)

df_test.drop(drop_columns_test,axis=1,inplace=True)
df_train.head()
#Machine Learning

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
X_train=df_train.drop('Survived',axis=1)

y_train=df_train['Survived']

X_test=df_test.drop('PassengerId',axis=1)
knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

knn.score(X_train,y_train)
logreg=LogisticRegression()

logreg.fit(X_train,y_train)

logreg.score(X_train,y_train)
rfc=RandomForestClassifier()

rfc.fit(X_train,y_train)

rfc.score(X_train,y_train)

y_pred=rfc.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred

    })



filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)