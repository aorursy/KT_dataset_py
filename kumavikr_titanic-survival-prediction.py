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
df= pd.read_csv('../input/Titanic_Train.csv')

df.head(10)
df.dtypes
import seaborn as sns

sns.heatmap((df.isnull()))
df["Embarked"].fillna("S", inplace = True) 
df['Embarked'] = pd.get_dummies(df,'Embarked')
df.dtypes
df = df.drop(['PassengerId', 'Name','Cabin','Ticket'],axis =1)
df['Sex'] = pd.get_dummies(df['Sex'],drop_first=True)
df.dtypes
sns.barplot(x = 'Sex',y = 'Survived',data = df)
sns.countplot(x='Survived',hue='Pclass',data=df)
df.isnull().sum()
df['Age'].fillna(df['Age'].mean(),inplace = True)
df.isnull().sum()
df.dtypes
from sklearn.model_selection import train_test_split

X = df.drop('Survived',axis =1)

y = df['Survived']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 101)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest