import pandas as pd

import numpy as np

import re

import operator
from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

from sklearn.ensemble import GradientBoostingClassifier
titanic_df = pd.read_csv("../input/train.csv")

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
#replacing gender column with numeric values for categorical column

numsex = {"male":1 ,"female" :2}

df_train['Sex'] = df_train['Sex'].replace(numsex)

df_train['Sex'] = pd.to_numeric(df_train['Sex'])

df_test['Sex'] = df_test['Sex'].replace(numsex)

df_test['Sex'] = pd.to_numeric(df_test['Sex'])
#replacing Embarked column with numeric values for categorical column

numembark = {"S":1 ,"C" :2, "Q":3}

df_train['Embarked'] = df_train['Embarked'].replace(numembark)

df_train['Embarked'] = pd.to_numeric(df_train['Embarked'])

df_test['Embarked'] = df_test['Embarked'].replace(numembark)

df_test['Embarked'] = pd.to_numeric(df_test['Embarked'])

#Replacing the nulls in the embarked column

df_train['Embarked'] = df_train['Embarked'].fillna(1)

df_train['Embarked'].describe()

df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Age'].mean())

df_test['Embarked'] = df_test['Embarked'].fillna(1)

df_test['Embarked'].describe()

df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Age'].mean())
# Secondly I want to fill in NULL Age values 

# To fill age with no data -> I want to find the median of ages in their class and gender

number_of_pclass = 3

number_of_gender = 2

if len(df_train.Age[ df_train.Age.isnull() ]) > 0:

    median_age = np.zeros([number_of_pclass, number_of_gender], float)

    for f in range(number_of_pclass):   # class

        for g in range(number_of_gender):     # gender                                     

            median_age[f, g] = df_train[ (df_train.Pclass == f+1) & (df_train.Sex == g) ]['Age'].dropna().median()

    for f in range(number_of_pclass):  

        for g in range(number_of_gender):                                          

            df_train.loc[ (df_train.Age.isnull()) & (df_train.Pclass == f+1) & (df_train.Sex == g), 'Age'] = median_age[f,g]



if len(df_test.Age[ df_test.Age.isnull() ]) > 0:

    median_age = np.zeros([number_of_pclass, number_of_gender], float)

    for f in range(number_of_pclass):   # class

        for g in range(number_of_gender):     # gender                                     

            median_age[f, g] = df_test[ (df_test.Pclass == f+1) & (df_test.Sex == g) ]['Age'].dropna().median()

    for f in range(number_of_pclass):  

        for g in range(number_of_gender):                                          

            df_test.loc[ (df_test.Age.isnull()) & (df_test.Pclass == f+1) & (df_test.Sex == g), 'Age'] = median_age[f,g]





df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())

df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())



#Check if the values got replaced#

df_train.describe()

df_test.describe()
# A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

    

# Get all the titles and print how often each one appear.

titles_train = df_train["Name"].apply(get_title)

print(pd.value_counts(titles_train))

titles_test = df_test["Name"].apply(get_title)

print(pd.value_counts(titles_test))





# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles_train[titles_train == k] = v

    titles_test[titles_test == k] = v



# Verify that we converted everything.

print(pd.value_counts(titles_train))

print(pd.value_counts(titles_test))



# Add in the title column.

df_train["Title"] = titles_train

df_train['Title'] = pd.to_numeric(df_train['Title'])

df_test["Title"] = titles_test

df_test['Title'] = pd.to_numeric(df_test['Title'])
# A dictionary mapping family name to id

family_id_mapping = {}



df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]

df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"]

df_train["NameLength"] = df_train["Name"].apply(lambda x: len(x))

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

family_ids_train = df_train.apply(get_family_id, axis=1)

family_ids_test = df_test.apply(get_family_id, axis=1)



# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.

family_ids_train[df_train["FamilySize"] < 3] = -1

family_ids_test[df_test["FamilySize"] < 3] = -1



# Print the count of each unique id.



df_train["FamilyId"] = family_ids_train

df_test["FamilyId"] = family_ids_test
predictors = ["Pclass", "Sex", "Age", "Fare", "FamilyId", "FamilySize", "Embarked", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [linear_model.LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]



full_predictions = []

for alg, predictors in algorithms:

    # Fit the algorithm using the full training data.

    alg.fit(df_train[predictors], df_train["Survived"])

    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.

    predictions = alg.predict_proba(df_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)
# The gradient boosting classifier generates better predictions, so we weight it higher.

predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0

predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({

    "PassengerId":df_test["PassengerId"],

    "Survived":predictions})

submission.to_csv("kaggle.csv", index=False)