# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



import operator 

import re 



# machine learning

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import cross_validation

from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import KFold

from sklearn.feature_selection import SelectKBest, f_classif
# get titanic & test csv files as a DataFrame

titanic_train = pd.read_csv("../input/train.csv")

titanic_test    = pd.read_csv("../input/test.csv")



# preview the data

titanic_train.head()
# describe the data 

titanic_train.describe()
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

titanic_train.describe()
print(titanic_train["Embarked"].unique())

print(titanic_train["Sex"].unique())
# Replace missing values in embarked columns and converting non numeric column to numeric columns 

titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')



# Converting male to 0 and female to 1, converting 'S' to 0, 'C' to 1 and 'Q' to 2 

titanic_train.loc[titanic_train["Sex"] == "male","Sex"] = 0

titanic_train.loc[titanic_train["Sex"] == "female","Sex"] = 1



titanic_train.loc[titanic_train["Embarked"] =='S',"Embarked"] = 0

titanic_train.loc[titanic_train["Embarked"] =='C',"Embarked"] = 1

titanic_train.loc[titanic_train["Embarked"] =='Q',"Embarked"] = 2



titanic_train.describe()
# Applying Logisitc Regression with Cross Validation 



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



alg = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)

accuracy = scores.mean()

print(accuracy)

# Accuracy on training data
# Processing of test data 





titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
# Testing on test data

alg = LogisticRegression(random_state=1)



alg.fit(titanic_train[predictors], titanic_train["Survived"])



predictions = alg.predict(titanic_test[predictors])

##submission = pd.DataFrame({

  ##      "PassengerId": titanic_test["PassengerId"],

    ##    "Survived": predictions

    ##})



##submission.to_csv('titanic.csv', index=False)
# Random Forest with Cross Validation 



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

kf = cross_validation.KFold(titanic_train.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=kf)

accuracy = scores.mean()

print(accuracy)
# New Features

titanic_train["FamilySize"] = titanic_train["SibSp"] + titanic_train["Parch"]



titanic_train["NameLength"] = titanic_train["Name"].apply(lambda x: len(x))
# Titles 



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



titles = titanic_train["Name"].apply(get_title)

print(pd.value_counts(titles))



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



print(pd.value_counts(titles))



titanic_train["Title"] = titles
# Family Groups



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



family_ids = titanic_train.apply(get_family_id, axis=1)

family_ids[titanic_train["FamilySize"] < 3] = -1

print(pd.value_counts(family_ids))



titanic_train["FamilyId"] = family_ids
# Finding Best Features



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic_train[predictors], titanic_train["Survived"])



scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()

predictors = ["Pclass", "Sex", "Fare", "Title"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

kf = cross_validation.KFold(titanic_train.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=kf)

print(scores.mean())
# Ensembling 



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]

kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    train_target = titanic_train["Survived"].iloc[train]

    full_test_predictions = []

    for alg, predictors in algorithms:

        alg.fit(titanic_train[predictors].iloc[train,:], train_target)

        test_predictions = alg.predict_proba(titanic_train[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1

    predictions.append(test_predictions)



predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == titanic_train["Survived"]]) / len(predictions)

print(accuracy)
# Changes in test data 



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

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
# prediction on test data



predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    alg.fit(titanic_train[predictors], titanic_train["Survived"])

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4



predictions[predictions <= .5] = 0

predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('titanic.csv', index=False)