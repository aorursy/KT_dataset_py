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
df.set_index('PassengerId',inplace=True)
df.isnull().sum()
df.drop(['Cabin'],axis=1)
df.describe()
df['Age'].plot.box()
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
dfn=pd.get_dummies(df)
df.dtypes
df.head()
modifydict={

    'Pclass':'object',

    'Age':'int64'

}

df=df.astype(modifydict)
df.dtypes
dfn=pd.get_dummies(df)
dfn
dfn.to_csv('cleaned.csv')