# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math 

import scipy.stats as sts



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
print( len(df.index))

df.describe(include='all').transpose()
sns.countplot(x='Survived',data=df)
sns.countplot(x='Survived',hue='Sex',data=df)
sns.countplot(x='Survived',hue='Pclass',data=df)
df.Age.hist()
df.Fare.plot.hist(bins=20,figsize=(10,5))
sts.mode(df.Fare)
df.info()
sns.countplot(x='SibSp',data=df)
df.isnull()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')
sns.boxplot(x='Pclass',y='Age',data=df)
df.drop('Cabin',axis=1,inplace=True)

df.head(5)
df.dropna(inplace=True)
df.isnull()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')
df.head()
pd.get_dummies(df['Sex'])
sex=pd.get_dummies(df['Sex'],drop_first=True)
embark=pd.get_dummies(df['Embarked'])

embark.head()
embark=pd.get_dummies(df['Embarked'],drop_first=True)

embark.head()
pcl=pd.get_dummies(df['Pclass'],drop_first=True)

pcl.head()
df=pd.concat([df,sex,embark,pcl],axis=1)
df.head(5)

df.drop(['Sex','Pclass','PassengerId','Embarked','Name','Ticket'],axis=1,inplace=True)
df.head()
x=df.drop(['Survived'],axis=1)

y=df['Survived']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression(max_iter=1000)

logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

classification_report(y_test,predictions)
confusion_matrix(y_test,predictions)
accuracy_score(y_test,predictions)