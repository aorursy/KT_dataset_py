import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

titanic=pd.read_csv('../input/train.csv')

titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex']=='male','Sex']=0

titanic.loc[titanic['Sex']=='female','Sex']=1

titanic['Embarked']=titanic['Embarked'].fillna('S')

titanic.loc[titanic['Embarked']=='S','Embarked']=0

titanic.loc[titanic['Embarked']=='C','Embarked']=1

titanic.loc[titanic['Embarked']=='Q','Embarked']=2

titanic.head()
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

x_titanic=titanic[predictors]

y_titanic=titanic['Survived']

 

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean())
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before

kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
# Generating a familysize column

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]



# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic=pd.read_csv('../input/train.csv')

titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex']=='male','Sex']=0

titanic.loc[titanic['Sex']=='female','Sex']=1

titanic['Embarked']=titanic['Embarked'].fillna('S')

titanic.loc[titanic['Embarked']=='S','Embarked']=0

titanic.loc[titanic['Embarked']=='C','Embarked']=1

titanic.loc[titanic['Embarked']=='Q','Embarked']=2

titanic.head()
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm with the default paramters

# n_estimators is the number of trees we want to make

# min_samples_split is the minimum number of rows we need to make a split

# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean())
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before

kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
# Generating a familysize column

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]



# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
import re

import pandas as pd



# A function to get the title from a name

def get_title(name):

    # Use a regular expression to search for a title  

    # Titles always consist of capital and lowercase letters, and end with a period

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it

    if title_search:

        return title_search.group(1)

    return ""



# Get all of the titles, and print how often each one occurs

titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))



# Map each title to an integer  

# Some titles are very rare, so they're compressed into the same codes as other titles

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Verify that we converted everything

print(pd.value_counts(titles))



# Add in the title column

titanic["Title"] = titles
import operator



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

family_ids = titanic.apply(get_family_id, axis=1)



# There are a lot of family IDs, so we'll compress all of the families with less than three members into one code

family_ids[titanic["FamilySize"] < 3] = -1



# Print the count of each unique ID

print(pd.value_counts(family_ids))



titanic["FamilyId"] = family_ids
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



# Get the raw p-values for each feature, and transform them from p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores  

# Do you see how "Pclass", "Sex", "Title", and "Fare" are the best features?

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()



# Pick only the four best features

predictors = ["Pclass", "Sex", "Fare", "Title"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)



scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import numpy as np



# The algorithms we want to ensemble

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



# Initialize the cross-validation folds

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



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
titanic_test=pd.read_csv('../input/test.csv')

# First, we'll add titles to the test set

titles = titanic_test["Name"].apply(get_title)

# We're adding the Dona title to the mapping, because it's in the test set, but not the training set

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles

# Check the counts of each unique title

print(pd.value_counts(titanic_test["Title"]))



# Now we add the family size column

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



# Now we can add family IDs

# We'll use the same IDs we used earlier

print(family_id_mapping)



family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids

titanic_test['NameLength']=titanic_test["Name"].apply(lambda x: len(x))
titanic_test['Age']=titanic_test['Age'].fillna(titanic['Age'].median())

titanic_test.loc[titanic_test['Sex']=='male','Sex']=0

titanic_test.loc[titanic_test['Sex']=='female','Sex']=1

titanic_test.loc[titanic_test['Embarked']=='S','Embarked']=0

titanic_test.loc[titanic_test['Embarked']=='C','Embarked']=1

titanic_test.loc[titanic_test['Embarked']=='Q','Embarked']=2

titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



titanic_test.head()

                     
algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    # Fit the algorithm using the full training data.

    alg.fit(titanic[predictors], titanic["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)



# The gradient boosting classifier generates better predictions, so we weight it higher

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0

predictions[predictions > .5] = 1

submission=predictions.astype(int)

submission=pd.DataFrame({'PassengerId':titanic_test['PassengerId'],'Survived':submission})    

submission.to_csv('submission.csv',index=False)