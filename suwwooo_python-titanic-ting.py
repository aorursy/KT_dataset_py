import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import os
os.getcwd()
os.chdir("/Users/abusharkhm/Downloads")
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode(), inplace=True)
lb = LabelEncoder()
train['Sex'] = lb.fit_transform(train['Sex'].astype(str))
lb2 = LabelEncoder()
train['Embarked'] = lb2.fit_transform(train['Embarked'].astype(str))
train_Y = train[['Survived']]
train_X = train[['Age', 'Pclass', 'SibSp', 'Fare','Sex', 'Embarked']]
test = test[['Age', 'Pclass', 'SibSp', 'Fare','Sex', 'Embarked']]
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
myTree = DecisionTreeClassifier()
myTree = myTree.fit(train_X, train_Y)
lb3 = LabelEncoder()
test['Sex'] = lb3.fit_transform(test['Sex'].astype(str))
lb4 = LabelEncoder()
test['Embarked'] = lb4.fit_transform(test['Embarked'].astype(str))
Pred = myTree.predict(test)
test_data = pd.read_csv("./test.csv").values
result = np.c_[test_data[:,0].astype(int), Pred.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('./res2.csv', index=False)