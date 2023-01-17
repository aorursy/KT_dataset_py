# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'../input/train.csv')
test = pd.read_csv(r'../input/test.csv')
label = pd.read_csv(r'../input/gender_submission.csv')
train.describe()
train.head()
train['Age'].describe()
train['Child'] = 0
train['Child'][train['Age'] < 18] = 1

train['Pclass1'] = 0
train['Pclass2'] = 0
train['Pclass3'] = 0

train['Pclass1'][train['Pclass'] == 1] = 1
train['Pclass2'][train['Pclass'] == 2] = 1
train['Pclass3'][train['Pclass'] == 3] = 1

train['Embarked'] = train['Embarked'].fillna('S')

train['Embarked_C'] = 0
train['Embarked_Q'] = 0
train['Embarked_S'] = 0

train['Embarked_C'][train['Embarked'] == 'C'] = 1
train['Embarked_Q'][train['Embarked'] == 'Q'] = 1
train['Embarked_S'][train['Embarked'] == 'S'] = 1

train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1

test['Age'] = test['Age'].fillna(int(test['Age'].mean()))

test['Child'] = 0
test['Child'][test['Age'] < 18] = 1

test['Pclass1'] = 0
test['Pclass2'] = 0
test['Pclass3'] = 0

test['Pclass1'][test['Pclass'] == 1] = 1
test['Pclass2'][test['Pclass'] == 2] = 1
test['Pclass3'][test['Pclass'] == 3] = 1

test['Embarked_C'] = 0
test['Embarked_Q'] = 0
test['Embarked_S'] = 0

test['Embarked_C'][test['Embarked'] == 'C'] = 1
test['Embarked_Q'][test['Embarked'] == 'Q'] = 1
test['Embarked_S'][test['Embarked'] == 'S'] = 1

test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1
feature_train = train[['Pclass1', 'Pclass2', 'Pclass3', 'Sex', 'Age', 'Child', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
feature_test = test[['Pclass1', 'Pclass2', 'Pclass3', 'Sex', 'Age', 'Child', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
label_train = train['Survived']
label_test = label['Survived']
clf = DecisionTreeClassifier(max_depth=1, random_state=0)
clf = clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
print(accuracy_score(label_test, pred))