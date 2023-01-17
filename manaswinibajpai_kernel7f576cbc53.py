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
df.drop(['Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

df.head()
dummies=pd.get_dummies(df.Sex)

df_new=pd.concat([df,dummies],axis='columns')

df_new.drop(['Sex'],axis='columns',inplace=True)

df_new.head()
df_new.Age=df_new.Age.fillna(df_new.Age.mean())
x=df_new.drop(['Survived'],axis='columns')

x.head()
y=df_new.Survived

y.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
train_x,val_x,train_y,val_y=train_test_split(x,y,test_size=0.3)

model=RandomForestClassifier(n_estimators=500)

#model=LogisticRegression()

model.fit(train_x,train_y)
model.score(val_x,val_y)