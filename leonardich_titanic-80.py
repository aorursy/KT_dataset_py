import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv("../input/train.csv", index_col=0)

train_data.head()
test_data_final = pd.read_csv("../input/test.csv", index_col=0)

test_data_final.head()
train_data.describe()
test_data_final.describe()
pd.isnull(train_data).sum()
age_avg_train = train_data["Age"].mean()

age_std_train = train_data["Age"].std()
train_data["Age"].fillna(np.random.randint(age_avg_train - age_std_train, 

                                           age_avg_train + age_std_train), inplace=True)

train_data["Embarked"].fillna("Q", inplace=True)
pd.isnull(test_data_final).sum()
age_avg_test = test_data_final["Age"].mean()

age_std_test = test_data_final["Age"].std()

test_data_final["Age"].fillna(np.random.randint(age_avg_test - age_std_test, 

                                           age_avg_test + age_std_test), inplace=True)

test_data_final["Fare"].fillna(test_data_final["Fare"].median(), inplace=True)
# create a new column to see if who was traveling alone

train_data["n_of_Family_members"] = train_data["SibSp"] + train_data["Parch"]
train_data["n_of_Family_members"].value_counts()
train_data[["n_of_Family_members", "Survived"]].groupby("n_of_Family_members").sum()
# create a new column to store the info about if person is alone or not 

# 1 represent person who was traveling alone, 0 otherwise

train_data["Alone"] = 1

train_data.loc[train_data["n_of_Family_members"] > 0, "Alone"] = 0
test_data_final["n_of_Family_members"] = test_data_final["SibSp"] + test_data_final["Parch"]

test_data_final["n_of_Family_members"].value_counts()
test_data_final["Alone"] = 1

test_data_final.loc[test_data_final["n_of_Family_members"] > 0, "Alone"] = 0
# create a new column for titles using regular expressions

train_data["Title"] = train_data["Name"].str.extract(' ([A-Za-z]+)\.')
train_data["Title"].value_counts()
# clean up and group

train_data["Title"].replace(["Dr", "Rev", "Col", "Major", "Lady", "Countess", 

                            "Sir", "Capt", "Jonkheer", "Don"], "Unusual", inplace=True)

train_data["Title"].replace(["Ms", "Mlle"], "Miss", inplace=True)

train_data["Title"].replace("Mme", "Mrs", inplace=True)
train_data["Title"].value_counts()
# categorical ==> numeric

train_data["Title"] = train_data["Title"].map({"Mr": 0, "Miss": 1,

                                                        "Mrs": 2, "Master": 3, "Unusual": 4}).astype(int)
test_data_final["Title"] = test_data_final["Name"].str.extract(' ([A-Za-z]+)\.')
test_data_final["Title"].value_counts()
test_data_final["Title"].replace(["Rev", "Col","Dona", "Dr"], "Unusual", inplace=True)

test_data_final["Title"].replace(["Ms"], "Miss", inplace=True)
test_data_final["Title"].value_counts()
test_data_final["Title"] = test_data_final["Title"].map({"Mr": 0, "Miss": 1,

                                                        "Mrs": 2, "Master": 3, "Unusual": 4}).astype(int)
train_data.head()
test_data_final.head()
# drop the data we do not need 

train_data.drop(["Ticket", "Cabin", "Name", "Parch", 

                 "SibSp", "n_of_Family_members"], axis=1, inplace=True)

test_data_final.drop(["Ticket", "Cabin", "Name", 

                      "Parch", "SibSp", "n_of_Family_members"], axis=1, inplace=True)
train_data.head()
test_data_final.head()
train_data["Sex"] = train_data["Sex"].map({"female":0, "male": 1}).astype(int)

train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_data_final["Sex"] = test_data_final["Sex"].map({"female":0, "male": 1}).astype(int)

test_data_final["Embarked"] = test_data_final["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_data["Fare"].describe()
train_data.loc[train_data["Fare"] <= 7.9104, "Fare"] = 0

train_data.loc[(train_data["Fare"] > 7.9104) & (train_data["Fare"] <= 14.4542), "Fare"] = 1

train_data.loc[(train_data["Fare"] > 14.4542) & (train_data["Fare"] <= 31), "Fare"] = 2

train_data.loc[(train_data["Fare"] > 31), "Fare"] = 3

train_data.head()
test_data_final["Fare"].describe()
test_data_final.loc[test_data_final["Fare"] <= 7.9104, "Fare"] = 0

test_data_final.loc[(test_data_final["Fare"] > 7.9104) & (test_data_final["Fare"] <= 14.4542), "Fare"] = 1

test_data_final.loc[(test_data_final["Fare"] > 14.4542) & (test_data_final["Fare"] <= 31), "Fare"] = 2

test_data_final.loc[(test_data_final["Fare"] > 31), "Fare"] = 3

test_data_final.head()
train_data["Age"].describe()
train_data.loc[train_data["Age"] <= 22, "Age"] = 0

train_data.loc[(train_data["Age"] > 22) & (train_data["Age"] <= 27), "Age"] = 1

train_data.loc[(train_data["Age"] > 27) & (train_data["Age"] <= 35), "Age"] = 2

train_data.loc[train_data["Age"] > 35, "Age"] = 3
train_data.head()
test_data_final["Age"].describe()
test_data_final.loc[test_data_final["Age"] <= 22, "Age"] = 0

test_data_final.loc[(test_data_final["Age"] > 22) & (test_data_final["Age"] <= 27), "Age"] = 1

test_data_final.loc[(test_data_final["Age"] > 27) & (test_data_final["Age"] <= 35), "Age"] = 2

test_data_final.loc[test_data_final["Age"] > 35, "Age"] = 3

test_data_final.head()
y = train_data["Survived"]

y.head()
X = train_data.drop(["Survived"], axis=1)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# parameters for GridSearchCV

param_grid = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],

              "max_leaf_nodes": [8, 9, 10, 11, 12, 13, 14, 15, 16]}
# look for the best parameters for RandomForestClassifier

rnd_clf = RandomForestClassifier()

grid_search = GridSearchCV(rnd_clf, param_grid, cv=5)

grid_search.fit(X_train, y_train)
grid_search.best_params_
# rnd_clf_tuned = grid_search.best_estimator_

rnd_clf_tuned = RandomForestClassifier(n_estimators=1000, bootstrap=True, max_depth=12, max_leaf_nodes=8)

rnd_clf_tuned.fit(X_train, y_train)

y_pred_rf = rnd_clf_tuned.predict(X_test)

accuracy_score(y_test, y_pred_rf)
# trying voting classifier

votting_clf = VotingClassifier(estimators=[('rf', rnd_clf_tuned)], voting="soft")
votting_clf.fit(X_train, y_train)

y_pred_votting = votting_clf.predict(X_test)

accuracy_score(y_test, y_pred_votting)
# voting does not perform better than rf, using rf

rnd_clf_tuned.fit(X, y)

y_pred_final = rnd_clf_tuned.predict(test_data_final)
# create submission file 

submission = pd.DataFrame()

submission["PassengerId"] = test_data_final.index

submission["Survived"] = y_pred_final

submission["Survived"].head()
submission.to_csv("submission.csv", index=False)