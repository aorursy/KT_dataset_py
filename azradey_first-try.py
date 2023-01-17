#libraries
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
#read train and test files
import os

os.path.realpath('.')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combined = [train, test]
train.head()
test.head()
train.describe()
train['Sex'] = train['Sex'].replace({"male" : 0})
train['Sex'] = train['Sex'].replace({"female" : 1})
train.head()
train.isnull().any()
test['Sex'] = test['Sex'].replace({"male" : 0})
test['Sex'] = test['Sex'].replace({"female" : 1})
test.head()
train[["Age"]] = train[['Age']].fillna(train[['Age']].median())
test[["Age"]] = test[['Age']].fillna(test[['Age']].median())
train.isnull().any()
train_df = train.drop(['Name','Embarked', 'Ticket', 'Cabin','Fare','PassengerId'], axis=1)
test_df = test.drop(['Name', 'Embarked', 'Ticket','Cabin','Fare'], axis=1)
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_prediction = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
Y_prediction
Y_prediction.shape
test.shape
df = pd.DataFrame({'PassengerId': test['PassengerId'].values,
                      'Survived': Y_prediction})
df.to_csv("submission.csv", index=False)
