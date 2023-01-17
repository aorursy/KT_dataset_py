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
import matplotlib.pyplot as plt
% matplotlib inline
import math
import seaborn as sns
df = pd.read_csv('../input/train.csv')
dft = pd.read_csv('../input/test.csv')
df.head()
sns.countplot(x='Survived', data=df)
sns.countplot(x='SibSp',hue='Survived', data=df)
df.isnull()
df.isnull().sum()
df.drop('Cabin', axis=1, inplace=True)
df.dropna(inplace=True)
sns.heatmap(df.isnull(), yticklabels=False)
df.isnull().sum()
df.Sex = pd.get_dummies(df['Sex'],drop_first=True)
df['Male']=pd.get_dummies(df['Sex'],drop_first=True)
df.head()
df.drop('Sex',axis=1, inplace=True)
df.drop(['Embarked'], axis=1, inplace=True)
df.head()
df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
x=df.drop(['Survived'], axis=1)
y=df.Survived
reg.fit(x, y)
dft = pd.read_csv('../input/test.csv')
passid=dft.PassengerId
dft.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
dft.head()
dft['Male']=pd.get_dummies(dft['Sex'],drop_first=True)
dft.drop('Sex',axis=1, inplace=True)
dft.isnull().sum()
dft['Age'] = dft['Age'].fillna(dft['Age'].median())
dft['Fare'] = dft['Fare'].fillna(dft['Fare'].median())

predictions = reg.predict(dft)
s=({"PassengerId":passid,"Survived":predictions})
submit=pd.DataFrame(data=s)
submit.to_csv('titanic.csv',index=False)
