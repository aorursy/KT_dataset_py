# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
print(train.shape)

print(test.shape)

print(gender_submission.shape)
print(train.info())

print()

print(test.info())

print()

print(gender_submission.info())
train.describe()
train.describe(include='O')
train.head()
test.head()
gender_submission.head()
train['Age'] = train['Age'].fillna(train['Age'].mean())

train['Embarked'] = train['Embarked'].fillna('S')



test['Age'] = test['Age'].fillna(test['Age'].mean())

test['Embarked'] = test['Embarked'].fillna('S')

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train.info()
test.info()
train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1

train['Embarked'][train['Embarked'] == 'S'] = 0

train['Embarked'][train['Embarked'] == 'C'] = 1

train['Embarked'][train['Embarked'] == 'Q'] = 2



train.head(10)
test['Sex'][test['Sex'] == 'male'] = 0

test['Sex'][test['Sex'] == 'female'] = 1

test['Embarked'][test['Embarked'] == 'S'] = 0

test['Embarked'][test['Embarked'] == 'C'] = 1

test['Embarked'][test['Embarked'] == 'Q'] = 2



test.head(10)
np.random.seed(0)

train_X = train[['Age', 'Sex', 'Pclass', 'Fare', 'SibSp']]

train_y = train['Survived']

test_X = test[['Age', 'Sex', 'Pclass', 'Fare', 'SibSp']]

test_y = gender_submission['Survived']



model = DecisionTreeClassifier()#決定木

model.fit(train_X, train_y)



print('決定木:{}'.format(model.score(test_X, test_y)))
print(model.predict(test_X))
PassengerId = np.array(test['PassengerId']).astype(int)



my_solution = pd.DataFrame(model.predict(test_X), PassengerId, columns = ['Survived'])



my_solution.to_csv('my_titanic_1.csv', index_label = ['PassengerId'])