import pandas as pd
titanic_training_data = pd.read_csv("../input/train.csv")
titanic_training_data.head(5)
titanic_training_data.describe()
titanic_training_data["Age"] = titanic_training_data["Age"].fillna(titanic_training_data["Age"].median())
titanic_training_data.loc[titanic_training_data["Sex"]=="male","Sex"] = 0
titanic_training_data.loc[titanic_training_data["Sex"]=="female","Sex"] = 1
titanic_training_data["Embarked"] = titanic_training_data["Embarked"].fillna("S")
titanic_training_data.loc[titanic_training_data["Embarked"]== "S","Embarked"] = 0
titanic_training_data.loc[titanic_training_data["Embarked"]== "C","Embarked"] = 1
titanic_training_data.loc[titanic_training_data["Embarked"]== "Q","Embarked"] = 2
titanic_training_data.head(10)
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.metrics import accuracy_score
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()
kf = KFold(titanic_training_data.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic_training_data[predictors].iloc[train,:])
    train_target = titanic_training_data["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic_training_data[predictors].iloc[test,:])
    predictions.append(test_predictions)


predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy_score(titanic_training_data["Survived"],predictions)
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = titanic_training_data["Name"].apply(get_title)
print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

print(pd.value_counts(titles))

titanic_training_data["Title"] = titles

titanic_training_data["FamilySize"] = titanic_training_data["SibSp"] + titanic_training_data["Parch"]

titanic_training_data["NameLength"] = titanic_training_data["Name"].apply(lambda x: len(x))

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

family_ids = titanic_training_data.apply(get_family_id, axis=1)

family_ids[titanic_training_data["FamilySize"] < 3] = -1

# Print the count of each unique id.
print(pd.value_counts(family_ids))

titanic_training_data["FamilyId"] = family_ids
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

%matplotlib inline


predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic_training_data[predictors], titanic_training_data["Survived"])

print(selector)
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
scores = cross_val_score(alg, titanic_training_data[predictors], titanic_training_data["Survived"], cv=3)
print(scores.mean())
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic_training_data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic_training_data["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic_training_data[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic_training_data[predictors].iloc[test,:].astype(float))[:,1]
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
accuracy = sum(predictions[predictions == titanic_training_data["Survived"]]) / len(predictions)
print(accuracy)
titanic_test_data = pd.read_csv("../input/test.csv")
titanic_test_data.head(5)
titanic_test_data.describe()
titanic_test_data["Age"] = titanic_test_data["Age"].fillna(titanic_training_data["Age"].median())
titanic_test_data.loc[titanic_test_data["Sex"]=="male","Sex"] = 0
titanic_test_data.loc[titanic_test_data["Sex"]=="female","Sex"] = 1
titanic_test_data["Embarked"] = titanic_test_data["Embarked"].fillna("S")
titanic_test_data.loc[titanic_test_data["Embarked"]=="S","Embarked"] = 0
titanic_test_data.loc[titanic_test_data["Embarked"]=="C","Embarked"] = 1
titanic_test_data.loc[titanic_test_data["Embarked"]=="Q","Embarked"] = 2
titanic_test_data["Fare"] = titanic_test_data["Fare"].fillna(titanic_training_data["Fare"].median())
titanic_test_data.describe()
titanic_test_data.head(5)

titles = titanic_test_data["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test_data["Title"] = titles
print(pd.value_counts(titanic_test_data["Title"]))
titanic_test_data["FamilySize"] = titanic_test_data["SibSp"] + titanic_test_data["Parch"]
titanic_test_data["NameLength"]=titanic_test_data["Name"].apply(lambda x:len(x))

family_ids = titanic_test_data.apply(get_family_id, axis=1)
family_ids[titanic_test_data["FamilySize"] < 3] = -1
titanic_test_data["FamilyId"] = family_ids

titanic_test_data.head(5)


predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic_training_data[predictors], titanic_training_data["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test_data[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": titanic_test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("titanic_medrah_solution.csv", index=False)