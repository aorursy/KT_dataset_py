import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

combine = [train, test]
train.head()
train.info()

print('_'*80)

print("Unique value in 'Sex': ", train["Sex"].unique())

print("Value counts: ", train["Sex"].value_counts())
train.describe()
train.describe(include=['O'])
train["family_size"] = float("NaN")

test["family_size"] = float("NaN")
for dataframe in combine:

    dataframe["Embarked"] = dataframe["Embarked"].fillna("S")

    dataframe["Age"] = dataframe["Age"].fillna(dataframe["Age"].median())

test.Fare = test.Fare.fillna(test["Fare"].median())
for dataframe in combine:

    # Convert the male and female groups to integer form

    dataframe['Sex'] = dataframe['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Convert the Embarked classes to integer form

    dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Or use commands like this

# train.loc[train["Sex"] == "male", "Sex"] = 0
# Create features

train["family_size"] = train["SibSp"] + train["Parch"] + 1

test["family_size"] = test["SibSp"] + test["Parch"] + 1
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["family_size", "Survived"]].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# Construct features and the target

features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size", "Embarked"]].values

features_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size", "Embarked"]].values

target = train["Survived"]
from sklearn import tree



# Train on a tree

decision_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

decision_tree = decision_tree.fit(features, target)

print(decision_tree.score(features, target))

print(decision_tree.feature_importances_)



# Make prediciton

prediction_dt = decision_tree.predict(features_test)

PassengerId =np.array(test["PassengerId"]).astype(int)

solution_dt = pd.DataFrame(prediction_dt, PassengerId, columns = ["Survived"])

solution_dt.to_csv("solution_dt.csv", index_label = ["PassengerId"])
from sklearn.ensemble import RandomForestClassifier



# Train on a tree

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

random_forest = forest.fit(features, target)

print(random_forest.score(features, target))

print(random_forest.feature_importances_)



# Make prediciton

prediction_rf = random_forest.predict(features_test)

solution_rf = pd.DataFrame(prediction_rf, PassengerId, columns = ["Survived"])

solution_rf.to_csv("solution_rf.csv", index_label = ["PassengerId"])
ID = np.arange(0,10)

age = np.arange(10,20)

people = pd.DataFrame(age, ID, columns = ["Age"])

people.to_csv("people.csv", index_label = ["ID"])
people = pd.DataFrame({"ID":ID, "Age":age})

people.to_csv("People.csv", index=False)