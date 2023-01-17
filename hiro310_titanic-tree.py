import numpy as np

import pandas as pd
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train['distinction'] = 'train'

test['distinction'] = 'test'
test['Survived'] = 99
all_data = pd.concat([train,test], axis=0).reset_index(drop=True)
all_data.head()
all_data.tail()
all_data = all_data.drop(["Name","Ticket","Embarked","Cabin"], axis=1)
all_data.head()
all_data["Sex"][all_data["Sex"] == "male"] = 0

all_data["Sex"][all_data["Sex"] == "female"] = 1
all_data.head()
all_data.isnull().sum()
all_data["Age"].fillna(all_data.Age.median(), inplace=True)

all_data["Fare"].fillna(all_data.Fare.median(), inplace=True)
all_data.isnull().sum()
train = all_data[all_data['distinction'] == "train"]

train = train.reset_index(drop=True)
train.head()
train.tail()
test = all_data[all_data['distinction'] == "test"]

test = test.reset_index(drop=True)
test.head()
test.tail()
train = train.drop(["distinction"], axis=1)

test = test.drop(["distinction"], axis=1)

test = test.drop(["Survived"], axis=1)
train.head()
test.head()
from sklearn import tree
train_object = train["Survived"].values
print(train_object)
train_explan = train[["Pclass","Sex","Age","Fare","SibSp","Parch"]].values
print(train_explan)
titanic_tree = tree.DecisionTreeClassifier()
titanic_tree = titanic_tree.fit(train_explan, train_object)
print(titanic_tree)
test_explan = test[["Pclass","Sex","Age","Fare","SibSp","Parch"]].values
print(test_explan)
prediction = titanic_tree.predict(test_explan)
print(prediction)
pas_id = np.array(test["PassengerId"]).astype(int)
print(pas_id)
answer = pd.DataFrame(prediction, pas_id, columns = ["Survived"])
print(answer)
answer.to_csv("answer.csv", index_label = ["PassengerId"])