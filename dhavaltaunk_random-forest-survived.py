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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
x_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y_train = train['Survived']
sns.countplot(y_train)
y_train.value_counts()
x_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
x_train['Sex'] = x_train['Sex'].replace(['male','female'],[1,0]) 
x_test['Sex'] = x_test['Sex'].replace(['male','female'],[1,0])
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_test['Age'] = x_test['Age'].fillna(x_train['Age'].mean())
x_train['Embarked'] = x_train['Embarked'].fillna('S')
x_test['Embarked'] = x_test['Embarked'].fillna('S')
x_train['Embarked'] = x_train['Embarked'].replace({'C','Q','S'},{0,1,2})
x_test['Embarked'] = x_test['Embarked'].replace({'C','Q','S'},{0,1,2})
x_train['Sex'] = x_train['Sex'].fillna(x_train['Sex'].mode())
x_test['Sex'] = x_test['Sex'].fillna(x_train['Sex'].mode())
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())
x_train.info()
X_train = np.array(x_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
Y_train = np.array(y_train)
X_test = np.array(x_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
regr = RandomForestClassifier(max_depth=8,max_features=None, n_estimators=50, min_samples_split=8)
regr.fit(X_train, Y_train)
regr.score(X_train,Y_train)
x_test.info()
output = pd.DataFrame()
output['PassengerId'] = test['PassengerId']
output['Survived'] = regr.predict(x_test)
output.to_csv('output.csv',index=False)

