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
df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.head() 
df.info()
df['Pclass'].value_counts().plot(kind='bar')
df['Survived'].value_counts().plot(kind='bar')
df.isna().sum()
df['Age'].hist()
df[df['Age'].notnull()].shape
df['Cabin_pre']=df['Cabin'].str[0]

df['Cabin_pre'][339]='O'
df['Cabin_pre']=df['Cabin_pre'].fillna(value='O')
df['Cabin_pre']=df['Cabin_pre'].replace(df.groupby('Cabin_pre')['Survived'].mean())

df.groupby('Cabin_pre')['Survived'].mean()
df['Embarked']=df['Embarked'].replace(df.groupby('Embarked')['Survived'].mean())

df.groupby('Embarked')['Survived'].mean()
df['Fare'].hist()
from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer()
df['Fare_log']=pt.fit_transform(df['Fare'].values.reshape(-1,1))
df['Fare_log'].hist()
df['Ticket_s']=df['Ticket'].str[0]

df['Ticket_s']=df['Ticket_s'].apply(lambda x:1 if x.isnumeric() else 0)

df['Ticket_s'].value_counts()
df['Ticket_l']=df['Ticket'].str.len()
df=df.drop(['Ticket','Fare'],axis=1)
df['Sex']=df['Sex'].replace({'male':1,'female':0})
df=df.drop(['PassengerId','Cabin'],axis=1)
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()
y=df[df['Age'].notnull()]['Age']

x=df[df['Age'].notnull()]
test=df[df['Age'].isnull()].drop(['Age','Name'],axis=1)
x=x.fillna(value=x['Embarked'].max())

x.shape
gbr.fit(x,y.values)
df['Embarked']=df['Embarked'].fillna(value=x['Embarked'].max())
df.loc[df[df['Age'].isnull()].index,'Age']=gbr.predict(test)
df.info()
name=df.pop('Name')
from sklearn.svm import SVC

model=SVC()
Y=df['Survived'].values

X=df.drop(['Survived'],axis=1)
from sklearn.model_selection import cross_val_score
cross_val_score(model,X,Y,cv=5)