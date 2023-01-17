# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import operator

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



def convert_enum_to_numbers_from_list(df, column, enums):

    value = 0

    for enum in enums:

        df.loc[df[column] == enum, column] = value

        value = value + 1

       

def convert_enum_to_numbers(df, column):

    convert_enum_to_numbers_from_list(df, column, df[column].unique())

    

def setMissingDataToMedian(df, column):

    df[column] = df[column].fillna(df[column].median())
titanic = pandas.read_csv("../input/train.csv")



# Print the first 5 rows of the dataframe.

print(titanic.describe()) # gives warning for NaNs
def repair_missing_data(df):

    setMissingDataToMedian(df, "Age")

    setMissingDataToMedian(df, "Fare")



repair_missing_data(titanic)

print(titanic.describe())
print(titanic["Sex"].unique()) # just male and female



def convert_sex_column(df):

    convert_enum_to_numbers(df, "Sex")



convert_sex_column(titanic)

print(titanic)
print(titanic["Embarked"].unique()) # S, C, Q or n/a



# shows that there are only 2 passengers (both survivors), so add to the most common class (S)

print(titanic[pandas.isnull(titanic['Embarked'])])



def convert_embarked_column(df):

    df["Embarked"] = df["Embarked"].fillna("S")

    convert_enum_to_numbers_from_list(df, "Embarked", ['S', 'C', 'Q'])



convert_embarked_column(titanic)

print(titanic)
# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm class

alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits every time we run this.

kf = KFold(n_splits=3)



predictions = []

for train, test in kf.split(titanic):

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = titanic[predictors].iloc[train, :]

    # The target we're using to train the algorithm.

    train_target = titanic["Survived"].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test, :])

    predictions.append(test_predictions)
# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > 0.5] = 1

predictions[predictions <= 0.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)



print(accuracy) # 78.34%
# Initialize our algorithm

alg = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean()) # 78.79%
titanic_test = pandas.read_csv("../input/test.csv")

repair_missing_data(titanic_test)

convert_sex_column(titanic_test)

convert_embarked_column(titanic_test)

print(titanic_test)
# Train the algorithm using all the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pandas.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("titanic.csv", index=False)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean()) # 78.56 %
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean()) # 81.59%
def add_new_features(df):

    # Generating a familysize column

    df["FamilySize"] = df["SibSp"] + df["Parch"]

    # The .apply method generates a new series

    df["NameLength"] = df["Name"].apply(lambda x: len(x))

    

add_new_features(titanic)

add_new_features(titanic_test)
# A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



def add_title_column(df):

    # Get all the titles and print how often each one occurs.

    titles = df["Name"].apply(get_title)

    print(pandas.value_counts(titles))



    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

    # This differs from the mapping at dataquest because I have grouped the rare French titles (Mlle and Mme) with their English equivalents,

    # and all nobility together (we already have a field for sex)

    title_mapping = {

        "Mr": 1,

        "Miss": 2,

        "Mlle": 2,

        "Ms": 2,

        "Mrs": 3,

        "Mme": 3,

        "Master": 4,

        "Dr": 5,

        "Rev": 6,

        "Major": 7,

        "Col": 7,

        "Capt": 7,

        "Don": 8,

        "Dona": 8,

        "Sir": 8,

        "Lady": 8,

        "Countess": 8,

        "Jonkheer": 8}

    for k,v in title_mapping.items():

        titles[titles == k] = v



    # Verify that we converted everything.

    print(pandas.value_counts(titles))



    # Add in the title column.

    df["Title"] = titles



add_title_column(titanic)

add_title_column(titanic_test)
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



def set_family_ids(df):

    family_id_mapping = {}

    # Get the family ids with the apply method

    family_ids = df.apply(get_family_id, axis=1)



    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.

    family_ids[titanic["FamilySize"] < 3] = -1



    # Print the count of each unique id.

    print(pandas.value_counts(family_ids))



    df["FamilyId"] = family_ids

    

set_family_ids(titanic)

set_family_ids(titanic_test)
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



# Pick only the best features.

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)



scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean()) # 83.73%
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



# The algorithms we want to ensemble.

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



def run_algorithms(train, test, test_df):

    full_predictions = []

    train_target = titanic["Survived"].iloc[train]

    for alg, predictors in algorithms:

        # Fit the algorithm on the training data.

        alg.fit(titanic[predictors].iloc[train,:], train_target)

        # Select and predict on the test fold.  

        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.

        full_predictions.append(alg.predict_proba(test_df[predictors].iloc[test,:].astype(float))[:,1])

    # The gradient boosting classifier generates better predictions, so we weight it higher.

    predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.

    predictions[predictions <= .5] = 0

    predictions[predictions > .5] = 1

    predictions = predictions.astype(int)

    return predictions



all_predictions = []

for train, test in kf.split(titanic):

    all_predictions.append(run_algorithms(train, test, titanic))



# Put all the predictions together into one array.

all_predictions = np.concatenate(all_predictions, axis=0)



# Compute accuracy by comparing to the training data.

accuracy = sum(all_predictions[all_predictions == titanic["Survived"]]) / len(all_predictions)

print(accuracy) # 82.04%
train = range(len(titanic))

test = range(len(titanic_test))

predictions = run_algorithms(train, test, titanic_test)



submission = pandas.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv("titanic2.csv", index=False)