# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.isnull()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex', data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False,color='green',bins=40)
train['Age'].hist(bins=30,color='red',alpha=0.3)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='orange',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
#function to replace null age value with average value



def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

            if Pclass == 1:

                return 37

            elif Pclass == 2:

                return 29

            else:

                return 24

    else:

        return Age
# applying function on the column age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.isnull().sum()
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
pd.get_dummies(train['Embarked'],drop_first=True).head()
sex = pd.get_dummies(train['Sex'],drop_first=True)

Embarked = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()
train = pd.concat([train,sex,Embarked],axis=1)
train.head()
train.drop('Survived',axis=1).head()
train['Survived'].head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),

                                                   train['Survived'], test_size=0.30,

                                                   random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test,prediction)
accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,prediction)

accuracy
prediction