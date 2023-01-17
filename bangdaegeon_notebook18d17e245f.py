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
train = train_data

test = test_data
train_y = train['Survived']

y = train_y
x = test['PassengerId']
pd.set_option('display.float_format', lambda x : '%.2f' %x)
train.isnull().sum()
test.isnull().sum()
train.replace({'male':0, 'female':1}, inplace=True)
test.replace({'male':0, 'female':1}, inplace=True)
train.groupby('Sex')['Age'].mean()
test.groupby('Sex')['Age'].mean()
train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'), inplace=True)
del train['Cabin']

del test['Cabin']
train['Embarked'] = train['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)

train['Embarked'] = train['Embarked'].fillna(0)

test['Embarked'] = test['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)

test['Embarked'] = test['Embarked'].fillna(0)
train["Family"] = train["Parch"] - train["SibSp"]

test["Family"] = test["Parch"] - test["SibSp"]
data = ['Pclass','Sex','Fare','Embarked','Age','Family']
train.corr()
X_train = train[data]
X_test = test[data]
# X_train['Age'] = train['Age'].fillna(train['Age'].mean())
# X_test['Age'] = test['Age'].fillna(test['Age'].mean())
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model
model.fit(X_train, y)
y_test = model.predict(X_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=1)

model.fit(X_train, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': x, 'Survived': predictions})

output.to_csv('titanic.csv', index=False)

print("Your submission was successfully saved!")