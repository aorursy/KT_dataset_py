# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape,test.shape
train.head()
# columns

col = train.columns

col
train.drop(['PassengerId','Name','Ticket'],axis =1,inplace =True)
train.describe()
train.isnull().sum()
train.Cabin.unique()
train.Cabin.str[0].unique() 
train.Cabin = train.Cabin.str[0]

train.Cabin = train.Cabin.fillna("N")

sns.factorplot('Cabin','Survived', data=train,size=4,aspect=3)
train = pd.concat([train,pd.get_dummies(train['Cabin'],prefix='Cabin')],axis=1)

train = train.drop('Cabin',1)
missing_embark = train[train['Embarked'].isnull()]

missing_embark
similar_embark = train [(train['Fare']<82.0)&(train['Fare']>78.0)& (train['Cabin_B']==1)&(train['Pclass']==1)]

similar_embark
train.Embarked = train.Embarked.fillna('C')

sns.countplot(x='Embarked',hue ='Survived',data = train)

train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
train = pd.concat([train,pd.get_dummies(train['Embarked'],prefix='Embarked')],axis=1)

train = train.drop('Embarked',1)
sns.factorplot('Pclass','Survived',data =train, size =3)
train = pd.concat([train,pd.get_dummies(train['Pclass'],prefix='Pclass')],axis=1)

train = train.drop('Pclass',1)
train[['Fare','Survived']].groupby(['Survived'],as_index=False).mean()
df = train[['Fare','Survived']].groupby(['Fare'],as_index=False).mean()

sns.regplot(x=df.Fare,y= df.Survived,color="g")
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
train = pd.concat([train,pd.get_dummies(train['Sex'],prefix='Sex')],axis=1)

train = train.drop('Sex',1)
train.columns
sns.countplot(x='SibSp',hue ='Survived',data = train)

train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean()
sns.countplot(x='Parch',hue ='Survived',data = train)

train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
train.Age.isnull().sum()