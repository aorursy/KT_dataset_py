from sklearn import cross_validation

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import re



titanic = pd.read_csv('../input/train.csv', header=0)

titanic_test = pd.read_csv('../input/test.csv', header=0)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Replace all the occurences of male with the number 0.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

# Replace all the occurences of female with the number 1.

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



# Repeat with test dataset

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2





# Generating a familysize column

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))



# A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Get all the titles

titles = titanic["Name"].apply(get_title)

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Add in the title column.

titanic["Title"] = titles



titles_test = titanic["Name"].apply(get_title)

for k,v in title_mapping.items():

    titles_test[titles_test == k] = v

titanic_test["Title"] = titles

    

    

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()
from sklearn.ensemble import GradientBoostingClassifier

# The algorithms we want to ensemble.

# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.

algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



# Initialize the cross validation folds

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)



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
titanic['Sex'] = titanic['Sex'].astype(int)

titanic['Embarked'] = titanic['Embarked'].astype(int)

titanic['Title'] = titanic['Title'].astype(int)
import xgboost as xgb


algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

    [xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



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

    #XGBoost

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
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]

train_X = titanic[predictors].as_matrix()

test_X = titanic_test[predictors].as_matrix()

train_y = titanic["Survived"]

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(train_X)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)



#print(gbm.feature_importances_)

xgb.plot_importance(gbm)



predictions = gbm.predict(test_X)

submission = pd.DataFrame({ 'PassengerId': titanic_test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)