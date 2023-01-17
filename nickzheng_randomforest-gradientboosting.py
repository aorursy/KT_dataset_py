# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

import re,operator

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.cross_validation import KFold
train = pd.read_csv("../input/train.csv" )

test = pd.read_csv("../input/test.csv")

print("\n\nTop of the training data:")

print(train.head())

print("\n\nSummary statistics of training data")

print(train.describe())



print("\n\nTop of the testing data:")

print(test.head())

print("\n\nSummary statistics of testing data")

print(test.describe())
train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())



#Handling Non-Numeric Columns

train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"] == "female", "Sex"] = 1  



train["Embarked"] = train["Embarked"].fillna("S")



train.loc[train["Embarked"] == "S", "Embarked"] = 0

train.loc[train["Embarked"] == "C", "Embarked"] = 1

train.loc[train["Embarked"] == "Q", "Embarked"] = 2



test.loc[test["Sex"] == "male", "Sex"] = 0

test.loc[test["Sex"] == "female", "Sex"] = 1  



test["Embarked"] = test["Embarked"].fillna("S")



test.loc[test["Embarked"] == "S", "Embarked"] = 0

test.loc[test["Embarked"] == "C", "Embarked"] = 1

test.loc[test["Embarked"] == "Q", "Embarked"] = 2
train["FamilySize"] = train["SibSp"] + train["Parch"]

test["FamilySize"] = test["SibSp"] + test["Parch"]



# The .apply method generates a new series

train["NameLength"] = train["Name"].apply(lambda x: len(x))

test["NameLength"] = test["Name"].apply(lambda x: len(x))
# Extracting The Passengers' Titles With A Regular Expression

# A function to get the title from a name

def get_title(name):

    # Use a regular expression to search for a title  

    # Titles always consist of capital and lowercase letters, and end with a period

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it

    if title_search:

        return title_search.group(1)

    return ""



titles = train["Name"].apply(get_title)

print(pd.value_counts(titles))
# Map each title to an integer  

# Some titles are very rare, so they're compressed into the same codes as other titles

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Verify that we converted everything

#  print(pandas.value_counts(titles))

# Add in the title column

train["Title"] = titles
# First, we'll add titles to the test set

titles = test["Name"].apply(get_title)

print(pd.value_counts(titles))

# We're adding the Dona title to the mapping, because it's in the test set, but not the training set

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

test["Title"] = titles
# Generating A Feature For Family Groups

# A dictionary mapping family name to ID

family_id_mapping = {}



# A function to get the ID for a particular row

def get_family_id(row):

    # Find the last name by splitting on a comma

    last_name = row["Name"].split(",")[0]

    # Create the family ID

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    # Look up the ID in the mapping

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            # Get the maximum ID from the mapping, and add 1 to it if we don't have an ID

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]
# Get the family IDs with the apply method

family_ids = train.apply(get_family_id, axis=1)

# There are a lot of family IDs, so we'll compress all of the families with less than three members into one code

family_ids[train["FamilySize"] < 3] = -1

# Print the count of each unique ID

#print(pd.value_counts(family_ids))

train["FamilyId"] = family_ids

print(train.head())

# Get the family IDs with the apply method

family_ids = test.apply(get_family_id, axis=1)

# There are a lot of family IDs, so we'll compress all of the families with less than three members into one code

family_ids[test["FamilySize"] < 3] = -1

test["FamilyId"] = family_ids

print(test.head())
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(train[predictors], train["Survived"])



scores = -np.log10(selector.pvalues_)



# Plot the scores  

# Do you see how "Pclass", "Sex", "Title", and "Fare" are the best features?

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()
# Pick only the four best features

#predictors = ["Pclass", "Sex", "Fare", "Title"]

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before

scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
#Making Predictions With Multiple Classifiers

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



# Initialize the cross-validation folds

kf = KFold(train.shape[0], n_folds=3, random_state=1)
predictions = []

for train_tmp, test_tmp in kf:

    train_target = train["Survived"].iloc[train_tmp]

    full_test_predictions = []

    # Make predictions for each algorithm on each fold

    for alg, predictors in algorithms:

        # Fit the algorithm on the training data

        alg.fit(train[predictors].iloc[train_tmp,:], train_target)

        # Select and predict on the test fold 

        # We need to use .astype(float) to convert the dataframe to all floats and avoid an sklearn error

        test_predictions = alg.predict_proba(train[predictors].iloc[test_tmp,:].astype(float))[:,1]

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

accuracy = sum(predictions[predictions == train["Survived"]]) / len(predictions)

print(accuracy)
#Predicting On The Test Set

#predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [

    [RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    #[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]

]



full_predictions = []

#print(test[predictors].iloc[:])

#print(type(test[predictors]))

for alg, predi in algorithms:

    # Fit the algorithm using the full training data.

    alg.fit(train[predi], train["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error

    predictions = alg.predict_proba(test[predi].astype(float))[:,1]

    #predicitons = alg.predict(test[predi])

    full_predictions.append(predictions)



# The gradient boosting classifier generates better predictions, so we weight it higher

#predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

#predictions[predictions <= .5] = 0

#predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv("kaggle.csv", index=False)