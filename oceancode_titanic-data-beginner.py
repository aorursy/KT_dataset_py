

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import tree



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df["Child"] = float('NaN')

train_df["Child"][train_df["Age"] < 18] = 1

train_df["Child"][train_df["Age"] >= 18] = 0



train_df[['Child', 'Survived']].groupby(['Child'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())



train_df["Sex"][train_df["Sex"] == "male"] = 0

train_df["Sex"][train_df["Sex"] == "female"] = 1





train_df["Embarked"] = train_df["Embarked"].fillna("S")

train_df["Embarked"][train_df["Embarked"] == "S"] = 0

train_df["Embarked"][train_df["Embarked"] == "C"] = 1

train_df["Embarked"][train_df["Embarked"] == "Q"] = 2

target_feature = train_df["Survived"].values

investigating_features = train_df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]].values
Tree_model = tree.DecisionTreeClassifier()

Tree_model = Tree_model.fit(investigating_features, target_feature)

Tree_model.feature_importances_
test_df.head()
test_df.info()
test_df.describe()
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())



test_df["Sex"][test_df["Sex"] == "male"] = 0

test_df["Sex"][test_df["Sex"] == "female"] = 1





test_df["Embarked"] = test_df["Embarked"].fillna("S")

test_df["Embarked"][test_df["Embarked"] == "S"] = 0

test_df["Embarked"][test_df["Embarked"] == "C"] = 1

test_df["Embarked"][test_df["Embarked"] == "Q"] = 2
investigating_features_test = test_df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]].values
predictions= Tree_model.predict(investigating_features_test)
PassengerId =np.array(test_df["PassengerId"]).astype(int)

solution_df = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])

solution_df.to_csv("solution_df.csv", index_label = ["PassengerId"])


