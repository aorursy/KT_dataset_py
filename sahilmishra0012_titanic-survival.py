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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train=train.drop(['PassengerId','Name','Ticket','Fare'],axis=1)
train.head()
train['Survived'].value_counts()
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['Age'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
train.isna().sum()
train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Cabin']=train['Cabin'].fillna('NONE')
train['Age']=train['Age'].fillna(train['Age'].mean())
y=train['Survived']
train=train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
train_data,test_data,y_train,y_test=train_test_split(train,y,stratify=y)
train_data.shape
test_data.shape
train.columns
train_data[Sex=='male']=1
train_data.Sex.value_counts()