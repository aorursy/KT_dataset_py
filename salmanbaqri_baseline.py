# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
sns.countplot(x='Survived', hue='Pclass', data=df_train)
sns.countplot(x='Survived', hue='Sex', data=df_train)
df_train.isnull().sum().sort_values(ascending=False) ###get missing values for each column
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train['Age'].isnull().sum()/len(df_train['Age'])
df_train.drop('Cabin',axis=1,inplace=True) ###dropping the Cabin column.
df_train.drop(df_train[pd.isnull(df_train['Embarked'])].index,inplace=True)
df_train.head()
#df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.')
df_train['Age'].fillna(1000,inplace=True)
df_train.isnull().sum().sort_values(ascending=False) ##No missing values now.
df_train.drop('PassengerId',axis=1,inplace=True)
df_train['Sex']=pd.Categorical(df_train['Sex'])

df_train['Embarked']=pd.Categorical(df_train['Embarked'])
##engineering new feature

df_train['FamilySize']=df_train['SibSp']+df_train['Parch']

df_train.head()
df_train.drop('SibSp',axis=1,inplace=True)

df_train.drop('Parch',axis=1,inplace=True)

df_train.head()
df_train.drop('Name', axis=1, inplace=True)

df_train.drop('Ticket', axis=1, inplace=True)

df_train.head()
df_train=pd.get_dummies(df_train,drop_first=True)

df_train.head()
df_raw = df_test.copy()
df_test['FamilySize']=df_test['SibSp']+df_test['Parch']

#df_test.drop('SibSp',axis=1,inplace=True)

#df_test.drop('Parch',axis=1,inplace=True)

#df_test.drop('Name',axis=1,inplace=True)

#df_test.drop('Ticket',axis=1,inplace=True)

to_be_dropped=['SibSp','Name','Parch','Ticket','Cabin','PassengerId']

df_test.drop(to_be_dropped,axis=1,inplace=True)

df_test['Age'].fillna(df_test['Age'].mean(),inplace=True)
df_test.isnull().sum().sort_values(ascending=False)
df_test['Fare'].fillna(df_test['Fare'].mean(),inplace=True)
df_test.isnull().sum().sort_values(ascending=False)
df_test['Sex']=pd.Categorical(df_test['Sex'])

df_test['Embarked']=pd.Categorical(df_test['Embarked'])
df_test=pd.get_dummies(df_test,drop_first=True)
X_train=df_train.drop('Survived',axis=1)

Y_train=df_train['Survived']

X_test =df_test.copy()
X_train.shape, Y_train.shape, X_test.shape
#Logistic Regression

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, Y_train)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round( clf.score(X_train, Y_train) * 100, 2)

print (str(acc_log_reg) + ' percent')
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics 

gnb=GaussianNB()

y_pred_gnb = gnb.fit(X_train, Y_train).predict(X_test)
y_pred_gnb
df_test.head()
submission = pd.DataFrame({

        "PassengerId": df_raw["PassengerId"],

        "Survived": y_pred_gnb

    })



submission.to_csv('submission.csv', index=False)