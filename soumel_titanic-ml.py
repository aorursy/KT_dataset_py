# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")
train.head()
train.info()
sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')
sns.boxplot(x='Pclass',y='Age',data=train)
def null_value(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        elif Pclass == 2:

            return 27

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(null_value,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Name','Embarked','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
train.head()
X = train.drop('Survived',axis=1)

y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))