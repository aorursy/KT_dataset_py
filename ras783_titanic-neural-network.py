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
import seaborn as sns

%matplotlib inline

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.info()
train.drop(['Cabin','PassengerId','Ticket','Name'],axis =1,inplace=True)
train = pd.concat([train,pd.get_dummies(train['Pclass'],prefix='Class')],axis=1)

train = pd.concat([train,pd.get_dummies(train['Embarked'],prefix='Embarked')],axis=1)

train = pd.concat([train,pd.get_dummies(train['Sex'],prefix='Sex',drop_first=True)],axis=1)

train.drop(['Pclass','Embarked','Sex'],axis=1,inplace=True)
train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(train['Age'].mean(),inplace=True)