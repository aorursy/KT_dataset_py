import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import re

import operator



from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, f_classif



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



titanic = pd.read_csv("../input/train.csv")



titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))





# A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Get all the titles and print how often each one occurs.

titles = titanic["Name"].apply(get_title)

#print(pd.value_counts(titles))



# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Verify that we converted everything.

#print(pandas.value_counts(titles))

titanic["Title"] = titles



# A dictionary mapping family name to id

family_id_mapping = {}



# A function to get the id given a row

def get_family_id(row):

    # Find the last name by splitting on a comma

    last_name = row["Name"].split(",")[0]

    # Create the family id

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    # Look up the id in the mapping

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            # Get the maximum id from the mapping and add one to it if we don't have an id

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]



# Get the family ids with the apply method

family_ids = titanic.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.

family_ids[titanic["FamilySize"] < 3] = -1

titanic["FamilyId"] = family_ids



#print(titanic)

#alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

#kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

#scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

#print(scores.mean())

#========================================================================



titanic_test = pd.read_csv("../input/test.csv")



titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



titanic_test["Pclass"] = titanic_test["Pclass"].fillna("2")

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())



#titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

#titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))



titles = titanic_test["Name"].apply(get_title)

# We're adding the Dona title to the mapping, because it's in the test set, but not the training set

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles



# Now, we add the family size column.

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



# Now we can add family ids.

# We'll use the same ids that we did earlier.

#print(family_id_mapping)



family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))



#========================================================================



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=50, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    # Fit the algorithm using the full training data.

    alg.fit(titanic[predictors], titanic["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)



# The gradient boosting classifier generates better predictions, so we weight it higher.

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions = (predictions >= 0.5).astype(int)



submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })

    

submission.to_csv("submission.csv", index=False)

submission.head()