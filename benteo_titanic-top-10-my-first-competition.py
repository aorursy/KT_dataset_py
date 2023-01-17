%matplotlib inline

# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random

from sklearn.metrics import mean_squared_error

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
# Load the training data in a dataframe

train = pd.read_csv("../input/train.csv")



# Load the test data in a dataframe

test = pd.read_csv("../input/test.csv")
# Display some data

train.head()
# Show the columns of the dataframe and their original type

for column in train.columns:

    print(column, train[column].dtype)
# Convert some columns into categorical data

train["Survived"] = train["Survived"].astype('category')

train["Pclass"] = train["Pclass"].astype('category')

test["Pclass"] = test["Pclass"].astype('category')

train.describe()
train["Survived"].value_counts().plot.pie(figsize=(4, 4),

                                          autopct='%.2f',

                                          title="Percentage of survivors",

                                          fontsize = 10)
train["Sex"].value_counts().plot.pie(figsize=(4, 4),

                                     autopct='%.2f',

                                     title="Percentage of Male and Female passengers",

                                     fontsize = 10)
sns.countplot(x="Survived", hue="Sex", data=train);
train["Pclass"].value_counts().plot.pie(figsize=(4, 4),

                                        autopct='%.2f',

                                        title="Percentage of passengers per Class",

                                        fontsize = 10)
sns.countplot(x="Survived", hue="Pclass", data=train);
def process_family(data):

    """ Aggregate the family size"""

    print("Processing family")

    data["familysize"] = data["SibSp"] + data["Parch"]

    print("    Done")

    return data



def process_ticket(data):

    """ Get further informations from the ticket number.

    Some passengers have the same ticket number. 

    We assume that it means they belong to the same group of people.

    """

    print("Processing ticket")

    data["ticketgroupsize"] = data.groupby("Ticket")["Ticket"].transform("count") - 1

    print("    Done")

    return data



def find_nan(data, feature, error=False):

    """ Look for missing values in a specific feature,

    count them and display the number. Raise an error 

    if asked. """

    if data[feature].isnull().values.any():

        print("    Missing values: ",

              data[feature].isnull().sum(),

              "over",

              len(data[feature].index),

              "(",

              data[feature].isnull().sum()/len(data[feature].index),

              "%)")

        if error:

            raise ValueError("NaN")

    return data[feature].isnull().values.any()



def process_names(data):

    """ Get further informations from the name title and put

    it into a new categorical data named type."""

    print("Processing Names")

    find_nan(data, "Name")

    # It is chosen to regroup titles by relevant categories.

    # Modifying that should have a big impact on the model results.

    name_dict = {"Capt":       "officer",

                 "Col":        "officer",

                 "Major":      "officer",

                 "Dr":         "officer",

                 "Rev":        "officer",

                 "Jonkheer":   "snob",

                 "Don":        "snob",

                 "Sir" :       "snob",

                 "the Countess":"snob",

                 "Dona":       "snob",

                 "Lady" :      "snob",

                 "Mme":        "married",

                 "Ms":         "married",

                 "Mrs" :       "married",

                 "Miss" :      "single",

                 "Mlle":       "single",

                 "Mr" :        "man",

                 "Master" :    "boy"

                }

    data['prefix'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    data['type'] = data['prefix'].map(name_dict)



    # Dummy encoding

    # Create one column for each value of the categorical data and assign a

    # one or zero value.

    # This is needed for sklearn that can only deal with numbers.

    column_dummies = pd.get_dummies(data['type'],prefix='type')

    data = pd.concat([data,column_dummies],axis=1)

    print("    Done")

    

    return data



def process_sex(data):

    """ Dummy encoding for the sex. 

    Map one to male and zero to female """

    print("Processing Sex")

    find_nan(data, "Sex", error=True)

    data['Sex'] = data['Sex'].map({'male':1,'female':0})

    print("    Done")

    

    return data



def process_age(data):

    """ Deal with missing age values. 

    A lot of data is missing in the age column.

    This is filled by a categorized median age.

    """

    print("Processing Age")

    find_nan(data, "Age")

    

    medianage = data.groupby(['Sex', 'Pclass', 'type'])["Age"].median()

    

    def fillna_age(row, medianage):

        age = medianage.loc[row["Sex"], row['Pclass'], row["type"]]

        return age



    data["Age"] = data.apply(lambda row : fillna_age(row, medianage) if np.isnan(row['Age']) else row['Age'], axis=1)

    find_nan(data, "Age")

    print("    Done")

    

    return data



def process_fare(data):

    """ Deal with missing fare values."""

    print("Processing Fare")

    find_nan(data, "Fare")

    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    print("    Done")

    

    return data



def process_embarked(data):

    """ Deal with missing embarked data and create

    the dummy encoding. """

    print("Processing Embarked")

    find_nan(data, "Embarked")

    # Find the most common occurence of the categorical data

    most_common = data['Embarked'].value_counts().index[0]

    # Replace NaN values with the most common occurence

    data['Embarked'].fillna(most_common, inplace=True)

    

    # dummy encoding

    # Create one column for each value of the categorical data

    column_dummies = pd.get_dummies(data['Embarked'],prefix='Embarked')

    data = pd.concat([data,column_dummies],axis=1)

    # Drop the now irrelevant column

    data.drop('Embarked',axis=1,inplace=True)

    print("    Done")

    

    return data



def process_cabin(data):

    """ Get the deck information from the cabin feature

    and deal with missing values. """

    print("Processing Cabin")

    find_nan(data, "Cabin")

    # Replace NaN values with U for unknown

    data['Cabin'].fillna("U", inplace=True)

    # Extract the deck information

    data['deck'] = data["Cabin"].map(lambda row: row[0])

    # dummy encoding

    # Create one column for each value of the categorical data

    column_dummies = pd.get_dummies(data['deck'],prefix='deck')

    data = pd.concat([data,column_dummies],axis=1)

    print("    Done")

    

    return data



def process_all(data):

    """ Process all the dataset features and return the dataset """

    data = process_family(data)

    data = process_ticket(data)

    data = process_sex(data)

    data = process_names(data)

    data = process_age(data)

    data = process_fare(data)

    data = process_embarked(data)

    data = process_cabin(data)

    return data



def write_results(data, model):

    """ Write results in the csv format for competition submission """

    with open("titanic.csv","w") as outfile:

        outfile.write("PassengerId,Survived\n")

        for passenger in data.index:

            line = str(data.at[passenger, "PassengerId"]) + "," + str(int(data.at[passenger, model])) + "\n"

            outfile.write(line)
# Load

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Merge & process

merged = train.append(test)

merged = process_all(merged)

# Split

train = pd.DataFrame(merged.head(len(train)))

test = pd.DataFrame(merged.iloc[len(train):])
train.columns
train["eval_random"] = [np.random.choice([0,1], p=[0.62, 0.38]) for passenger in train.index]



rmse_random = np.sqrt(mean_squared_error(train["Survived"], train["eval_random"]))

print(rmse_random)
def richwoman(passenger, data):

    """ If you are a female from the first class, you survive. """

    if data.at[passenger, "Sex"] == "female" and data.at[passenger, "Pclass"] == 1:

        return 1

    else:

        return 0

train["eval_richwoman"] = [richwoman(passenger, train) for passenger in train.index]



rmse_richwoman = np.sqrt(mean_squared_error(train["Survived"], train["eval_richwoman"]))

print(rmse_richwoman)
features_names = ["Fare",

                  "SibSp",

                  "Parch",

                  "familysize",

                  "ticketgroupsize",

                  "Pclass",

                  "Sex", 

                  "Age",

                  "Embarked_C", 

                  "Embarked_Q",

                  "Embarked_S",

                  "type_boy",

                  "type_officer",

                  "type_married",

                  "type_single",

                  "type_snob",

                  "type_man",

                  "deck_A",

                 "deck_B",

                 "deck_C",

                 "deck_D",

                 "deck_E",

                 "deck_F",

                 "deck_G",

                 "deck_U"]



# We select here all the features above.

features = train[features_names] 

# The target is to predict the survived category

target = train["Survived"]

# The tree is a decision tree classifier.

my_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

# The tree is fitted to the train data.

my_tree = my_tree.fit(features, target)



# Sklearn has a built in score that we can look at and from that score tune the tree parameters.

# Here we look at the score on the train data so beware of overfitting.

print("Score of tree on train data: ", my_tree.score(features, target))



# Use the tree to evaluate its answer on the train data.

train["eval_tree"] = my_tree.predict(features)

# Look at the RMSE score on the train data.

rmse_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tree"]))

print("RMSE:", rmse_tree)
features_imp = pd.DataFrame()

features_imp['feature'] = features_names

features_imp['importance'] = my_tree.feature_importances_

features_imp.sort_values(by=['importance'], ascending=True, inplace=True)

features_imp.set_index('feature', inplace=True)

features_imp.plot(kind='barh', figsize=(20, 20))
test["eval_tree"] = my_tree.predict(test[features_names].values)



write_results(test, "eval_tree")
# The forest will have 50 trees 

# and the max number of features by trees is the square root of the total features number

my_forest = RandomForestClassifier(n_estimators=50, max_features='sqrt')

my_forest = my_forest.fit(features, target)
features_imp = pd.DataFrame()

features_imp['feature'] = features_names

features_imp['importance'] = my_forest.feature_importances_

features_imp.sort_values(by=['importance'], ascending=True, inplace=True)

features_imp.set_index('feature', inplace=True)

features_imp.plot(kind='barh', figsize=(20, 20))
print("Score of forest on train data: ", my_forest.score(features, target))



train["eval_forest"] = my_forest.predict(features)



rmse_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_forest"]))

print("RMSE:", rmse_tree)
test["eval_forest"] = my_forest.predict(test[features_names].values)



write_results(test, "eval_forest")
def tune_forest(features, targets):

    """ Find the best parameters for the random forest """

    #parameter_grid = {

    #             'max_depth' : [5, 6, 7],

    #             'n_estimators': [20],

    #             'max_features': ['sqrt', 'auto', 'log2'],

    #             'min_samples_split': [2, 5, 10, 15],

    #             'min_samples_leaf': [3, 10],

    #             'bootstrap': [True, False],

    #             }

    parameter_grid = None

    parameter_grid = {

                 'max_depth' : [8, 10, 12],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }

    forest = RandomForestClassifier(n_jobs=2)



    grid_search = GridSearchCV(forest,

                               scoring='accuracy',

                               param_grid=parameter_grid,

                               cv=3,

                               n_jobs=2,

                               verbose=1)



    grid_search.fit(features, targets)

    model = grid_search.best_estimator_

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best estimator: {}'.format(grid_search.best_estimator_))

    

    return model, parameters
model, parameters = tune_forest(features, target)

train["eval_tuned_forest"] = model.predict(train[features_names].values)

rmse_tuned_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tuned_forest"]))

print("RMSE:", rmse_tuned_tree)
test["eval_tuned_forest"] = model.predict(test[features_names].values)



write_results(test, "eval_tuned_forest")
model = xgb.XGBClassifier()

model.fit(features, target)



print("Score of tree on train data: ", model.score(features, target))



train["eval_xgb_tree"] = model.predict(features)



rmse_xgb_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_xgb_tree"]))

print("RMSE:", rmse_xgb_tree, "1-RMSE:", 1.0-rmse_xgb_tree)
features_imp = pd.DataFrame()

features_imp['feature'] = features_names

features_imp['importance'] = model.feature_importances_

features_imp.sort_values(by=['importance'], ascending=True, inplace=True)

features_imp.set_index('feature', inplace=True)

features_imp.plot(kind='barh', figsize=(20, 20))
test["eval_xgb_tree"] = model.predict(test[features_names])



write_results(test, "eval_xgb_tree")
def tune_xgb_tree(features, targets):

    parameter_grid = {

                 'max_depth' : [7, 8, 9],

                 'max_delta_step': [1],

                 'n_estimators': [20, 40, 60, 80],

                 'colsample_bylevel': [0.8, 0.9, 1.0],

                 'colsample_bytree': [0.6, 0.8, 1.0],

                 'subsample': [0.3, 0.4, 0.5, 0.6],

                 }

    xgb_model = xgb.XGBClassifier()

    print(xgb_model.get_params().keys())



    grid_search = GridSearchCV(xgb_model,

                               scoring='accuracy',

                               param_grid=parameter_grid,

                               cv=3,

                               n_jobs=2,

                               verbose=1)



    grid_search.fit(features, targets)

    model = grid_search.best_estimator_

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best estimator: {}'.format(grid_search.best_estimator_))

    

    return model, parameters
model, parameters = tune_xgb_tree(features, target)
train["eval_tuned_xgb_tree"] = model.predict(train[features_names])

rmse_tuned_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tuned_xgb_tree"]))

print("RMSE:", rmse_tuned_tree, "1-RMSE:", 1.0-rmse_tuned_tree)
parameters
features_imp = pd.DataFrame()

features_imp['feature'] = features_names

features_imp['importance'] = model.feature_importances_

features_imp.sort_values(by=['importance'], ascending=True, inplace=True)

features_imp.set_index('feature', inplace=True)

features_imp.plot(kind='barh', figsize=(20, 20))
test["eval_tuned_xgb_tree"] = model.predict(test[features_names])



write_results(test, "eval_tuned_xgb_tree")