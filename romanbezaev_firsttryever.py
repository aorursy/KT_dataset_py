import numpy as np

import pandas as pd 

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
train_data.Age.fillna(train_data.Age.median(), inplace=True)

train_data.Fare.fillna(train_data.Fare.median(), inplace=True)

X_train = pd.get_dummies(train_data[features])

y_train = train_data.Survived
from sklearn.model_selection import GridSearchCV

r_tree = RandomForestClassifier()

parameters = {'n_estimators' : range(30,52,10),

              'max_depth' : range(9,22,3)}

grid_tree = GridSearchCV(r_tree, parameters, cv=5)
grid_tree.fit(X_train, y_train)
test_data.Age.fillna(test_data.Age.median(), inplace=True)

test_data.Fare.fillna(test_data.Fare.median(), inplace=True)
X_test = pd.get_dummies(test_data[features])

best_tree = grid_tree.best_estimator_
y_predicted = best_tree.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': y_predicted})

output.to_csv('submission.csv', index=False)