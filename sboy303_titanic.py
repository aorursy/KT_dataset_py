

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

titanic=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

titanic.head(10)
titanic.isna().sum()
df = titanic.copy()
df.groupby(['Survived']).count()
sns.countplot(df['Survived'])

fig = plt.gcf()

fig.set_size_inches(5,5)

plt.title('surviver')
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()
titanic.isna().sum()
sns.heatmap(titanic.isna(),yticklabels=False,cmap="ocean")
df = titanic.copy()
df = df.dropna()
sns.heatmap(df.isna(),yticklabels=False,cmap="viridis",cbar=False)
df.head()
sex = pd.get_dummies(df['Sex'],drop_first=True)

sex.head()
embark = pd.get_dummies(df['Embarked'],drop_first=True)

embark.head()
Pc = pd.get_dummies(df['Pclass'],drop_first=True)

Pc.head()
df = pd.concat([df,sex,embark,Pc],axis=1)

df.head()
df.drop(['Sex','PassengerId','Name','Ticket'],axis=1,inplace=True)
df.head()
df.drop('Pclass',axis=1,inplace=True)

df.head()
train=['Age','Q','male','Fare','S','SibSp','Parch',2,3]

X=df[train]

Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
lmodel=LogisticRegression()
lmodel.fit(X_train,Y_train)
prediction=lmodel.predict(X_test)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



classification_report(Y_test,prediction)
confusion_matrix(Y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,prediction)