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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 
data = pd.read_csv('../input/titanic/train.csv')
data.head()
data.tail()
data.info()
data.describe()
data.isnull().sum()
sns.heatmap(data.isnull(),cbar=False, yticklabels=False)
sns.pairplot(data)
sns.countplot(x='Survived', data= data)
sns.countplot(x='Survived', data=data, hue='Sex')
sns.countplot(x='Survived', data=data, hue='Pclass')
sns.distplot(data['Age'].dropna())
sns.distplot(data['Age'].dropna(), kde=False, bins=40)
sns.countplot(x='SibSp',data=data)
data['Fare'].hist(bins=30, figsize=(8,4))
sns.boxplot(x='Pclass', y='Age', data=data)
def age_(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        elif Pclass ==2:

            return 27

        else:

            return 24

        

    else:

        return Age
data['Age'] = data[['Age', 'Pclass']].apply(age_, axis=1)
sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
data.drop('Cabin', axis=1, inplace=True)
sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
data.isnull().sum()
data.dropna(inplace=True, axis=0)
data.head()
data.info()
data.isnull().sum()
sex = pd.get_dummies(data['Sex'], drop_first=True)

sex
embrkd = pd.get_dummies(data['Embarked'], drop_first=True)

embrkd

train_data = data.copy()
train_data.head()
train_data = pd.concat([train_data,sex,embrkd], axis=1, ignore_index=False)
train_data.head()
train_data.drop(['Sex','Embarked','Name','Ticket','PassengerId'], inplace=True, axis=1)
train_data.isnull().sum()
train_data.head(2)
X= train_data.drop('Survived', axis=1)

y = train_data['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.30, random_state=86)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
pred
from sklearn.metrics import confusion_matrix, classification_report, classification, accuracy_score
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))