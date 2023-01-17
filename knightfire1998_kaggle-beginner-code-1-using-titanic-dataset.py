# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data["Sex"] =pd.get_dummies(train_data["Sex"], drop_first = True) 
test_data["Sex"] =pd.get_dummies(test_data["Sex"], drop_first = True) 
train_data.info()
train_data.drop("Cabin", axis = 1 , inplace = True)
train_data.drop("Name", axis = 1 , inplace = True)
test_data.drop("Cabin", axis = 1 , inplace = True)
test_data.drop("Name", axis = 1 , inplace = True)
train_data.info()

train_data['Embarked'].fillna('S',  inplace = True)
test_data['Age'].fillna((test_data['Age'].median()), inplace = True)
train_data['Age'].fillna((train_data['Age'].median()), inplace = True)

train_data['Embarked'] = train_data['Embarked'].map({'Q':2, 'S':1, 'C':0})
test_data['Embarked'] = test_data['Embarked'].map({'Q':2, 'S':1, 'C':0})
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.countplot('Survived',data=train_data, hue = 'Sex')
train_data.info()
sns.heatmap(train_data.isnull())
train_data['Age']=train_data['Age'].astype('int64')
test_data['Age']= test_data['Age'].astype('int64')
train_data['Embarked']=train_data['Embarked'].astype('int64')
test_data['Embarked']= test_data['Embarked'].astype('int64')


train_data.tail()
X = train_data.drop(['Survived', 'Ticket', 'Fare'], axis = 1)
y = train_data['Survived']
X_test = test_data.drop(['Ticket', 'Fare'], axis = 1)
sns.heatmap(X.isnull())

X.info()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

model.score(X, y)
random_forest_accuracy = model.score(X, y)
random_forest_accuracy

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
model.score(X, y)
random_forest_accuracy = model.score(X, y)
random_forest_accuracy
