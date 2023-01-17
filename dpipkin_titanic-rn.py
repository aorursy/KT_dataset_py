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
df = pd.read_csv('../input/train.csv', index_col='PassengerId')
df.sample(10)
df['Survived'].sum() / len(df)
1 - _
len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)]) / \
  len(df[df['Sex'] == 'male'])
len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)]) / \
  len(df[df['Sex'] == 'female'])
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'])
x_train.head()
x_train = pd.get_dummies(x_train, columns=['Sex'])
x_train = x_train[['Sex_female','Sex_male']]
x_train.sample(10)
clf.fit(x_train, y_train)
x_test = pd.get_dummies(x_test, columns=['Sex'])[['Sex_female', 'Sex_male']]
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
test_data = pd.read_csv('../input/test.csv', index_col='PassengerId')
x = pd.get_dummies(test_data, columns=['Sex'])[['Sex_female', 'Sex_male']]
y = clf.predict(x)
y = pd.Series(y, index=x.index)
y.head()
y.name = 'Survived'
pd.read_csv('../input/gender_submission.csv').sample(10)
y.to_csv('random_forest_gender.csv', header=True)