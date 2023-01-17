%matplotlib inline

from __future__ import division

import pandas

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

import matplotlib.pyplot as plt

import re

import operator
titanic = pandas.read_csv("../input/train.csv")

print(titanic.head(5))

print(titanic.describe())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic.describe()
print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna('S')

titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0

titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1

titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2
#setting up the linear regression classifier with kfolds

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []

for train, test in kf:

    train_predictors = (titanic[predictors].iloc[train,:])

    train_target = titanic["Survived"].iloc[train]

    alg.fit(train_predictors, train_target)

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = sum(map(lambda x, y: x == y, predictions, titanic["Survived"]))/len(titanic["Survived"])

accuracy
alg = LogisticRegression(random_state=1)

scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())
titanic_test = pandas.read_csv("../input/train.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test.loc[titanic_test["Embarked"] == 'S', "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == 'C', "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == 'Q', "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())
alg = LogisticRegression(random_state=1)

alg.fit(titanic[predictors], titanic["Survived"])

predictions = alg.predict(titanic_test[predictors])

submission = pandas.DataFrame({

       "PassengerId": titanic_test["PassengerId"],    "Survived": predictions  })

submission.to_csv("kaggle.csv", index=False)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



titles = titanic["Name"].apply(get_title)

print(pandas.value_counts(titles))



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 

                 "Major": 7, "Col": 7, "Capt": 7,

                 "Mlle": 8, "Mme": 8, 

                 "Don": 9, 

                 "Lady": 10, "Countess": 10, "Jonkheer": 10, 

                 "Sir": 9, 

                 "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



print(pandas.value_counts(titles))



titanic["Title"] = titles
family_id_mapping = {}



def get_family_id(row):

    last_name = row["Name"].split(",")[0]

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]



family_ids = titanic.apply(get_family_id, axis=1)



family_ids[titanic["FamilySize"] < 3] = -1



print(pandas.value_counts(family_ids))



titanic["FamilyId"] = family_ids
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



scores = -np.log10(selector.pvalues_)



plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()
predictors = ["Pclass", "Sex", "Fare", "Title"]



alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)

scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())


algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]

    full_test_predictions = []

    for alg, predictors in algorithms:

        alg.fit(titanic[predictors].iloc[train,:], train_target)

        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1

    predictions.append(test_predictions)



predictions = np.concatenate(predictions, axis=0)



accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 

                 "Major": 7, "Col": 7, "Capt": 7,

                 "Mlle": 8, "Mme": 8, 

                 "Don": 9, "Sir": 9,

                 "Lady": 10, "Countess": 10, "Jonkheer": 10, "Dona": 10,  

                 "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles



print(pandas.value_counts(titanic_test["Title"]))
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1



print(pandas.value_counts(family_ids))



titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    alg.fit(titanic[predictors], titanic["Survived"])

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)



predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4



predictions[predictions > 0.5] = 1

predictions[predictions <= 0.5] = 0

predictions = predictions.astype(int)



submission = pandas.DataFrame({"PassengerId": titanic_test["PassengerId"],

                               "Survived": predictions})

submission.to_csv("kaggle_gradientboosting.csv",index=False)