import os

import pandas as pd



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print('Shape of train dataset: {}'.format(train.shape))

print('Shape of test dataset: {}'.format(test.shape))
train.drop(['Name', 'Ticket', 'Cabin'],axis = 1,inplace = True)

train.head()
test.drop(['Name', 'Ticket', 'Cabin'],axis = 1,inplace = True)

test.head()
print(train.isnull().values.any())

train.isnull().sum()
print(train['Age'].value_counts())

train['Age'] = train['Age'].fillna(24)

train.isnull().sum()
print(train['Embarked'].value_counts())

train['Embarked'] = train['Embarked'].fillna('S')

train.isnull().sum()
print(test['Age'].value_counts())

test['Age'] = test['Age'].fillna(24)

print(test['Fare'].value_counts())

test['Fare'] = test['Fare'].fillna(7.75)

test.isnull().sum()
# Categorical data to numerical data

train.loc[train['Sex'] == 'male', 'Sex'] =  1

train.loc[train['Sex'] == 'female', 'Sex'] = 0

train.loc[train['Embarked'] == 'S', 'Embarked'] =  0

train.loc[train['Embarked'] == 'Q', 'Embarked'] = 1

train.loc[train['Embarked'] == 'C', 'Embarked'] =  2

print(train.head())



test.loc[test['Sex'] == 'male', 'Sex'] =  1

test.loc[test['Sex'] == 'female', 'Sex'] = 0

test.loc[test['Embarked'] == 'S', 'Embarked'] =  0

test.loc[test['Embarked'] == 'Q', 'Embarked'] = 1

test.loc[test['Embarked'] == 'C', 'Embarked'] =  2

print(test.head())
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



train_y = train['Survived']

train_x = train.drop('Survived', axis=1)

model = LogisticRegression()

model.fit(train_x, train_y)

pred = model.predict(train_x)

metrics.accuracy_score(pred, train_y)
import time



timestamp = int(round(time.time() * 1000))



pred = model.predict(test)

output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})

output.to_csv("submission_" + str(timestamp) + ".csv",index = False)