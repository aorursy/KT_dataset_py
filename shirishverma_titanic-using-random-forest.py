# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# uploading files

train_1 = pd.read_csv("../input/train.csv")

test_1 = pd.read_csv("../input/test.csv")
# understanding files_1

train_1.head(20)
# understanding files_2

test_1.head()
# finding out missing values

train_1.info()

print("----------------")

test_1.info()
# checking which one has dominance

train_1["Embarked"].value_counts()
# filling missing embarked with 'S'

train_1["Embarked"].fillna('S', inplace = True)
# verifying

train_1["Embarked"].value_counts()
# visualizing distribution of age

train_1['Age'].hist(bins = 10)

test_1['Age'].hist(bins = 10)

plt.show()
# filling missing age values

train_1["Age"].fillna(train_1["Age"].median(), inplace = True)

test_1["Age"].fillna(test_1["Age"].median(), inplace = True)
# check

train_1.describe()
# dropping columns

train_1.drop(['PassengerId'], axis = 1, inplace = True)
train_1.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
test_1.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
# check

train_1.info()

print("----------------")

test_1.info()
plt.bar(train_1['Pclass'], train_1['Fare'], color = "blue")

plt.show()
# finding class whose fare is missing

test_1['Pclass'][test_1['Fare'].isnull() == True]
test_1["Fare"][test_1["Pclass"] == 3].mode()
test_1["Fare"].fillna(7.75, inplace = True)
test_1.info()
train_1.info()
train_1.head()
import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



sex_train = train_1[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean()

sns.barplot(x = "Sex", y = "Survived", data = sex_train, order = ["male", "female"])
train_1.head()
train_1["Person"] = train_1["Sex"].copy()
train_1["Person"][train_1["Age"] < 18] = "child"
train_1.head(10)
test_1["Person"] = test_1["Sex"].copy()

test_1["Person"][test_1["Age"] < 18] = "child"
train_1.drop(["Sex"], axis = 1, inplace = True)

test_1.drop(["Sex"], axis = 1, inplace = True)
person_survived = train_1[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_survived, order=["male", "female", "child"])
person_dummies_train = pd.get_dummies(train_1["Person"])
person_dummies_train.drop(["male"], axis = 1, inplace = True)

train_1 = train_1.join(person_dummies_train)



person_dummies_test = pd.get_dummies(test_1["Person"])

person_dummies_test.drop(["male"], axis = 1, inplace = True)

test_1 = test_1.join(person_dummies_test)
train_1.head()
train_1.drop(["Person"], axis = 1, inplace = True)

test_1.drop(["Person"], axis = 1, inplace = True)
train_1.head()
sns.factorplot("Embarked", "Survived", data = train_1, size = 4, aspect = 3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot("Embarked", data = train_1, ax = axis1)

sns.countplot(x='Survived', hue="Embarked", data=train_1, order=[1,0], ax=axis2)

embark_perc = train_1[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
test_1.head()
embark_dummies_train = pd.get_dummies(train_1["Embarked"])

embark_dummies_train.drop(["S"], axis = 1, inplace = True)



embark_dummies_test = pd.get_dummies(test_1["Embarked"])

embark_dummies_test.drop(["S"], axis = 1, inplace = True)
train_1 = train_1.join(embark_dummies_train)

test_1 = test_1.join(embark_dummies_test)



train_1.drop(["Embarked"], axis = 1, inplace = True)

test_1.drop(["Embarked"], axis = 1, inplace = True)
train_1.head(10)
train_1["Family"] = train_1["SibSp"] + train_1["Parch"]

test_1["Family"] = test_1["SibSp"] + test_1["Parch"]
train_1["Family"][train_1["Family"] == 0] = 0

train_1["Family"][train_1["Family"] > 0] = 1



test_1["Family"][test_1["Family"] == 0] = 0

test_1["Family"][test_1["Family"] > 0] = 1



family_only = train_1[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x = "Family", y = "Survived", data = family_only, order = [1,0])
train_1.drop(["SibSp", "Parch"], axis = 1, inplace = True)

test_1.drop(["SibSp", "Parch"], axis = 1, inplace = True)
train_1.head()
sns.factorplot("Pclass", "Survived", data = train_1, size = 4, aspect = 3)
class_dummies_train = pd.get_dummies(train_1["Pclass"])

class_dummies_train.drop([3], axis = 1, inplace = True)



class_dummies_test = pd.get_dummies(test_1["Pclass"])

class_dummies_test.drop([3], axis = 1, inplace = True)



train_1 = train_1.join(class_dummies_train)

test_1 = test_1.join(class_dummies_test)



train_1.drop(["Pclass"], axis = 1, inplace = True)

test_1.drop(["Pclass"], axis = 1, inplace = True)
train_1.head()
from sklearn.ensemble import RandomForestClassifier



target = train_1["Survived"]

features = train_1[["Age", "Fare", "child", "female", "Family", "C", "Q", 1, 2]]



forest_1 = RandomForestClassifier(min_samples_leaf = 10, n_estimators = 100)

my_forest = forest_1.fit(features, target)
print(my_forest.feature_importances_)
print(my_forest.score(features, target))
test_1.head()
test_features = test_1[["Age", "Fare", "child", "female", "Family", "C", "Q", 1, 2]]

pred = my_forest.predict(test_features)
submission = pd.DataFrame({

        "PassengerId": test_1["PassengerId"],

        "Survived": pred

    })

submission.to_csv('my_solution_8.csv', index=False)