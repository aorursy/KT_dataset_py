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
from pandas import Series, DataFrame

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
titanic = pd.read_csv("../input/train.csv")
print(titanic.head(5))
print(titanic.describe())
titanic['Age']= titanic['Age'].fillna(titanic['Age'].median())
print(titanic.describe())
titanic.loc[titanic['Sex']== 'male', 'Sex'] = 0

titanic.loc[titanic['Sex']== 'female', 'Sex'] = 1
titanic['Embarked'].unique()
titanic['Embarked']= titanic['Embarked'].fillna('S')

titanic['Embarked'].unique()
titanic.loc[titanic['Embarked']== 'S', 'Embarked'] = 0

titanic.loc[titanic['Embarked']== 'C', 'Embarked'] = 1

titanic.loc[titanic['Embarked']== 'Q', 'Embarked'] = 2

titanic['Embarked'].unique()
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold



# the columns

predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

linreg= LinearRegression()

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []

for train, test in kf:

    train_predictors = (titanic[predictors].iloc[train,:])

    train_target = titanic['Survived'].iloc[train]

    linreg.fit(train_predictors, train_target)

    test_predictions = linreg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions
# The predictions are in three different NumPy arrays. Concatenate into one array



predictions= np.concatenate(predictions, axis=0)

predictions[predictions> 0.5] = 1

predictions[predictions<= 0.5] = 0

accuracy= sum(predictions[predictions==titanic['Survived']])/ len(predictions)
print(accuracy)
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(logreg, titanic[predictors], titanic['Survived'], cv=3)

print(scores.mean())
titanic_test= pd.read_csv("../input/test.csv")
titanic_test.describe()
# The age has to be the exact same value we used to replace the missing ages in the training set

titanic_test['Age']= titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test.loc[titanic_test['Sex']== 'male', 'Sex'] = 0

titanic_test.loc[titanic_test['Sex']== 'female', 'Sex'] = 1
titanic_test['Embarked']= titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked']== 'S', 'Embarked'] = 0

titanic_test.loc[titanic_test['Embarked']== 'C', 'Embarked'] = 1

titanic_test.loc[titanic_test['Embarked']== 'Q', 'Embarked'] = 2
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier
predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(rf, titanic[predictors], titanic['Survived'], cv=kf)

print(scores.mean())
rf = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(rf, titanic[predictors], titanic['Survived'], cv=kf)

print(scores.mean())
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']



titanic['NameLen'] = titanic['Name'].apply(lambda x: len(x))
import re



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
# To create this feature, we'll concatenate each passenger's last name with FamilySize to get a unique family ID. 

# Then we'll be able to assign a code to each person based on their family ID.



import operator



family_id_mapping = {}



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
titanic.describe()
from sklearn.feature_selection import SelectKBest, f_classif



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLen"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



# Get the raw p-values for each feature, and transform them from p-values into scores

scores = -np.log10(selector.pvalues_)
# Plot the scores  

# "Pclass", "Sex", "Title", and "Fare" are the best features

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()
# Pick only the four best features

predictors = ["Pclass", "Sex", "Fare", "Title"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []

for train, test in kf:

    train_predictors = (titanic[predictors].iloc[train,:])

    train_target = titanic['Survived'].iloc[train]

    linreg.fit(train_predictors, train_target)

    test_predictions = linreg.predict(titanic[predictors].iloc[test,:])

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1

    predictions.append(test_predictions)
submission = DataFrame({'PassengerId': titanic_test['PassengerId'],

                        'Survived': predictions})