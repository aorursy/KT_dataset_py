# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train = df_train.drop(columns = ['PassengerId','Name','Ticket'])
df_train['Sex'] = pd.get_dummies(df_train['Sex']).drop(columns = ['female'])
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_train = df_train.drop(columns = 'Embarked').merge(pd.get_dummies(df_train['Embarked'], prefix = 'Embarked'),left_index = True, right_index = True)
df_train['Cabin'] = df_train['Cabin'].fillna('Z').astype(str).str[0]
df_train = df_train.drop(columns = 'Cabin').merge(pd.get_dummies(df_train['Cabin'], prefix = 'Cabin'),left_index = True, right_index = True)
df_train = df_train.drop(columns = 'Pclass').merge(pd.get_dummies(df_train['Pclass'], prefix = 'Pclass'),left_index = True, right_index = True)
df_train.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_train.drop(columns = ['Survived', 'Cabin_Z','Pclass_3'])) #scaler.fit_transform(df_train.drop(columns = ['Survived'])) # 
y = df_train['Survived'].to_numpy()
from sklearn.linear_model import LogisticRegression

#Train model
clf = LogisticRegression(random_state=0, max_iter = 100, penalty = 'none').fit(X, y)

np.abs((clf.predict(X) - y)).sum(), np.abs((clf.predict(X) - y)).mean()
