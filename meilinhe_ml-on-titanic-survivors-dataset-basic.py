## basic machine learning ##

#read in files

import numpy as np

import pandas as pd

titanic = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")

#cleansing

for df in [titanic, titanic_test]:

    df.loc[df["Sex"] == "male", "Sex"] = 0

    df.loc[df["Sex"] == "female", "Sex"] = 1

    df["Embarked"] = df["Embarked"].fillna("S")

    ports = list(df["Embarked"].unique())

    for i,port in enumerate(ports):

        df.loc[df["Embarked"]==port, "Embarked"] = i

    for col in ["Age", "Fare"]:

        df[col] = df[col].fillna(titanic[col].median())

    print(df.describe())
#train model & make predictions

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold



features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lr = LinearRegression()

kf = KFold(3, random_state=1)

predictions = []

for train, test in kf.split(titanic):

    train_feat = (titanic[features].iloc[train,:])

    train_tar = titanic["Survived"].iloc[train]

    lr.fit(train_feat, train_tar)

    test_feat = lr.predict(titanic[features].iloc[test,:])

    predictions.append(test_feat)
#test

# Concatenate predictions into single array

preds = np.concatenate(predictions, axis=0)

preds[preds > .5] = 1

preds[preds <=.5] = 0

matched = 0

for i, row in titanic.iterrows():

    if row["Survived"] == preds[i]:

        matched += 1

accuracy = matched/titanic.shape[0]

print(accuracy)
#prediction

predictions = lr.predict(titanic_test[features])

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })
# improve model

##  add features

####    family size

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

####    name length

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

####    title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""

titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v

print(pd.value_counts(titles))

titanic["Title"] = titles

####    family "id"

import operator

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

print(pd.value_counts(family_ids))

titanic["FamilyId"] = family_ids
# select best features

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform them from p-values into scores

scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()

# observation: "Pclass", "Sex", "Fare", and "Title" are four best features

predictors_four_best = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation

# The algorithms we want to ensemble

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



# Initialize the cross-validation folds

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]

    full_test_predictions = []

    # Make predictions for each algorithm on each fold

    for alg, predictors in algorithms:

        # Fit the algorithm on the training data

        alg.fit(titanic[predictors].iloc[train,:], train_target)

        # Select and predict on the test fold 

        # We need to use .astype(float) to convert the dataframe to all floats and avoid an sklearn error

        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    # Use a simple ensembling scheme&#8212;just average the predictions to get the final classification

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1

    predictions.append(test_predictions)



# Put all the predictions together into one array

predictions = np.concatenate(predictions, axis=0)



# Compute accuracy by comparing to the training data

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
# add features for test set

titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles

print(pd.value_counts(titanic_test["Title"]))

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids
# make new predictions

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
submission.to_csv("kaggle.csv", index=False)