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
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_test.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df_train.isnull(),cmap='YlGnBu')
sns.countplot('Survived',data=df_train)
sns.countplot('Survived',hue='Sex',data=df_train)
sns.countplot('Pclass',hue='Survived',data=df_train)
sns.set_style('whitegrid')
sns.distplot(df_train['Age'],kde=False,bins=30)
sns.countplot(x='SibSp',data=df_train)
df_train['Fare'].hist(bins=40,figsize=(8,4))
df_train.Cabin.isnull().value_counts()
df_train=df_train.drop('Cabin',axis=1)
df_test=df_test.drop('Cabin',axis=1)
df_train.head()
df_train.Age.isnull().value_counts()
df_test.Age.isnull().value_counts()
df_train.groupby('Pclass').mean().Age
df_test.groupby('Pclass').mean().Age
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train)
def age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 29
        else:
            return 25
    else:
        return age
    
df_train['Age'] = df_train[['Age','Pclass']].apply(age,axis=1)
sns.heatmap(df_train.isnull(),cmap='YlGnBu')
#Sex = pd.get_dummies(df_train['Sex'],drop_first=True)
#Embarked = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train=pd.get_dummies(df_train,columns=['Sex','Embarked'],drop_first=True)
df_train.head()
df_train.drop(['Name','Ticket','Survived','PassengerId'],axis=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter = 500000)
logmodel.fit(df_train.drop(['Name','Ticket','Survived'],axis=1),df_train['Survived'])
prediction=logmodel.predict(df_test.drop(['Age','Cabin','Name','Ticket'],axis=1))


