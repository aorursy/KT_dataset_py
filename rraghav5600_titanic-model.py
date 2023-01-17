from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.columns
train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + train['Parch']
f_file=test['PassengerId']

train=train.drop(['Cabin','PassengerId','Name', 'Ticket', 'SibSp', 'Parch'],axis=1)

test=test.drop(['Cabin','PassengerId','Name', 'Ticket', 'SibSp', 'Parch'],axis=1)

# train.dropna(inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)

train.dropna(inplace=True)

test['Age'].fillna(test['Age'].mean(), inplace=True)

# test.dropna(inplace=True)
my_label = LabelEncoder()

train['Sex'] = my_label.fit_transform(train['Sex'])

train['Embarked'] = my_label.fit_transform(train['Embarked'])

test['Sex'] = my_label.fit_transform(test['Sex'])

test['Embarked'] = my_label.fit_transform(test['Embarked'])

train.head()
train.head()
test.head()
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(train.drop(['Survived'], axis=1), train['Survived'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score



rf = RandomForestClassifier(random_state=1)

rf.fit(train_X, train_y)

prediction = rf.predict(val_X)

print(mean_absolute_error(val_y, prediction), accuracy_score(val_y, prediction))
from xgboost import XGBClassifier



XGB = XGBClassifier()

XGB.fit(train_X, train_y)

prediction = XGB.predict(val_X)

print(mean_absolute_error(val_y, prediction), accuracy_score(val_y, prediction))
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
vals = rf.predict(test)

file = pd.DataFrame({'PassengerId':f_file, 'Survived':vals})

file.to_csv('submission_rf.csv', index = False)

file.head()
vals = XGB.predict(test)

file = pd.DataFrame({'PassengerId':f_file, 'Survived':vals})

file.to_csv('submission_xgb.csv', index = False)

file.head()