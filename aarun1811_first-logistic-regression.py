# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

# Import the linear regression class

from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross validation

from sklearn.cross_validation import KFold



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/train.csv")

titanic['Age'].fillna(titanic['Age'].median(), inplace = True)

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic.loc[titanic['Embarked'].isnull(), 'Embarked'] = 'S'

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0

titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1

titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []

for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = titanic["Survived"].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
import numpy as np

from sklearn import metrics



# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0



accuracy = metrics.accuracy_score(titanic['Survived'], predictions)
print(accuracy)
titanic_test = pd.read_csv("../input/test.csv")

titanic_test['Age'].fillna(titanic['Age'].median(), inplace = True)



titanic_test.loc[titanic_test['Sex'] == 'male','Sex'] = 0

titanic_test.loc[titanic_test['Sex'] == 'female','Sex'] = 1



titanic_test.loc[titanic_test['Embarked'].isnull(), 'Embarked'] = 'S'



titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0

titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1

titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2



titanic_test.loc[titanic_test['Fare'].isnull(), 'Fare'] = titanic_test['Fare'].median()
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

alg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm with the default paramters

# n_estimators is the number of trees we want to make

# min_samples_split is the minimum number of rows we need to make a split

# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
# Generating a familysize column

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]



# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

import re



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

print(pd.value_counts(titles))



# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Verify that we converted everything.

print(pd.value_counts(titles))



# Add in the title column.

titanic["Title"] = titles
import operator



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



# Print the count of each unique id.

print(pd.value_counts(family_ids))



titanic["FamilyId"] = family_ids
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()



# Pick only the four best features.

predictors = ["Pclass", "Sex", "Fare", "Title"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np



# The algorithms we want to ensemble.

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



# Initialize the cross validation folds

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]

    full_test_predictions = []

    # Make predictions for each algorithm on each fold

    for alg, predictors in algorithms:

        # Fit the algorithm on the training data.

        alg.fit(titanic[predictors].iloc[train,:], train_target)

        # Select and predict on the test fold.  

        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.

        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    # Use a simple ensembling scheme -- just average the predictions to get the final classification.

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1

    predictions.append(test_predictions)



# Put all the predictions together into one array.

predictions = np.concatenate(predictions, axis=0)



# Compute accuracy by comparing to the training data.

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
titles = titanic_test["Name"].apply(get_title)

# We're adding the Dona title to the mapping, because it's in the test set, but not the training set

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles

# Check the counts of each unique title.

print(pd.value_counts(titanic_test["Title"]))



# Now, we add the family size column.

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



# Now we can add family ids.

# We'll use the same ids that we did earlier.

print(family_id_mapping)



family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

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

predictions[predictions <= .5] = 0

predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })
submission.to_csv("kaggle.csv", index=False)