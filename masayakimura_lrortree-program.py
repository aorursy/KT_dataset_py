import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train.shape)
print(train.isnull().sum())
print(test.isnull().sum())

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train.head(10)
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train = pd.get_dummies(train, columns=['Embarked'])

train.head(10)
test['Age'] = test['Age'].fillna(test['Age'].median())
test.loc[test['Sex'] == 'male','Sex'] = 0
test.loc[test['Sex'] == 'female','Sex'] = 1
test = pd.get_dummies(test, columns=['Embarked'])
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

test.head(10)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

target = train['Survived'].values #目的変数
features_one = train[['Pclass','Sex','Age','Embarked_C','Embarked_Q','Embarked_S']].values #説明変数
test_features = test[['Pclass','Sex','Age','Embarked_C','Embarked_Q','Embarked_S']].values

#決定木
#my_tree_one = DecisionTreeClassifier(max_depth=4)
#my_tree_one = my_tree_one.fit(features_one, target)
#my_prediction = my_tree_one.predict(test_features)

#ロジスティック回帰
logisticregression = LogisticRegression()
logisticregression = logisticregression.fit(features_one, target)
my_prediction = logisticregression.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'])
my_solution.to_csv('my_solution1.csv', index_label=['PassengerId'])