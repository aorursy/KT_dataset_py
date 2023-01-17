# Import the packages

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from scipy.stats import gaussian_kde

import random

import numpy as np

%matplotlib inline

plt.rcParams["figure.figsize"] = 10, 6 
# Read csv

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

combined = pd.concat((train, test))

combined = combined.reset_index()

combined.drop("index", axis=1, inplace=True)

print(train.shape, test.shape, combined.shape)
train.head()
combined.count()
combined.drop("Survived", axis=1, inplace=True)

combined.count()
survived = train[train["Survived"] == 1].Survived.value_counts()

dead = train[train["Survived"] == 0].Survived.value_counts()

plt.bar([0, 1], [dead, survived], align="center")

plt.xticks([0, 1], ["Dead", "Survived"])
survived_gender = train[train["Survived"] == 1].Sex.value_counts()

dead_gender = train[train["Survived"] == 0].Sex.value_counts()

df = pd.DataFrame([dead_gender, survived_gender])

df.index = ["Dead", "Survived"]

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
n, bins, patches = plt.hist([train[train["Survived"] == 1].Age.dropna(),

                             train[train["Survived"] == 0].Age.dropna()],

                            stacked=True, edgecolor="k", bins=80)

plt.legend(patches, ("Survived", "Dead"), loc="best")

plt.show()
sns.kdeplot(train[train["Survived"] == 1].Age.dropna(), label="Survived", shade=True, clip=[0., 80.])

sns.kdeplot(train[train["Survived"] == 0].Age.dropna(), label="Dead", shade=True, clip=[0., 80.])

plt.xlim(0., 80.)

plt.ylim(0., 0.035)

plt.show()
n, bins, patches = plt.hist([combined[combined["Sex"] == "male"].Age.dropna(),

                             combined[combined["Sex"] == "female"].Age.dropna()],

                            stacked=True, edgecolor="k", bins=80)

plt.legend(patches, ("male", "female"), loc="best")

plt.show()
class_1 = train[train["Pclass"] == 1].Survived.value_counts()

class_2 = train[train["Pclass"] == 2].Survived.value_counts()

class_3 = train[train["Pclass"] == 3].Survived.value_counts()

df = pd.DataFrame([class_1, class_2, class_3])

df.index = ["Class 1", "Class 2", "Class 3"]

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
class_1 = combined[combined["Pclass"] == 1].Sex.value_counts()

class_2 = combined[combined["Pclass"] == 2].Sex.value_counts()

class_3 = combined[combined["Pclass"] == 3].Sex.value_counts()

df = pd.DataFrame([class_1, class_2, class_3])

df.index = ["Class 1", "Class 2", "Class 3"]

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
sns.kdeplot(combined[combined["Pclass"] == 1].Age.dropna(), label="Class 1", shade=True, clip=[0., 80.])

sns.kdeplot(combined[combined["Pclass"] == 2].Age.dropna(), label="Class 2", shade=True, clip=[0., 80.])

sns.kdeplot(combined[combined["Pclass"] == 3].Age.dropna(), label="Class 3", shade=True, clip=[0., 80.])

plt.xlim(0., 80.)

plt.ylim(0., 0.045)

plt.show()
emb_S = train[train["Embarked"] == "S"].Survived.value_counts()

emb_C = train[train["Embarked"] == "C"].Survived.value_counts()

emb_Q = train[train["Embarked"] == "Q"].Survived.value_counts()

df = pd.DataFrame([emb_S, emb_C, emb_Q])

df.index = ["S", "C", "Q"]

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
emb_S_class = combined[combined["Embarked"] == "S"].Pclass.value_counts()

emb_C_class = combined[combined["Embarked"] == "C"].Pclass.value_counts()

emb_Q_class = combined[combined["Embarked"] == "Q"].Pclass.value_counts()

df = pd.DataFrame([emb_S_class, emb_C_class, emb_Q_class])

df.index = ["S", "C", "Q"]

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
# get unique identifiers at the start of the tickets

index_first_string = ~combined["Ticket"].str.split(" ").str.get(0).str.isdigit()

combined[index_first_string]["Ticket"].str.split(" ").str.get(0).unique()
combined["Cabin"].unique()
decks = list(combined["Cabin"].str.get(0).dropna().unique())

decks_survive = []

for deck in decks:

    decks_survive.append(train[train["Cabin"].str.get(0) == deck].Survived.value_counts())

df = pd.DataFrame(decks_survive)

df.index = decks

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()

# Also plot the Nan values

decks_survive.append(train[train["Cabin"].isnull()].Survived.value_counts())

decks.append("U")

df = pd.DataFrame(decks_survive)

df.index = decks

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
decks = list(combined["Cabin"].str.get(0).dropna().unique())

decks_class = []

for deck in decks:

    decks_class.append(combined[combined["Cabin"].str.get(0) == deck].Pclass.value_counts())

df = pd.DataFrame(decks_class)

df.index = decks

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()

# Also plot the Nan values

decks_class.append(combined[combined["Cabin"].isnull()].Pclass.value_counts())

decks.append("U")

df = pd.DataFrame(decks_class)

df.index = decks

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
train.SibSp.value_counts(), train.Parch.value_counts()
siblings = sorted(train.SibSp.unique())[1:]

sibs = []

for sib in siblings:

    sibs.append(train[train["SibSp"] == sib].Survived.value_counts())

df = pd.DataFrame(sibs)

df.index = siblings

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
family = sorted(train.Parch.unique())[1:]

families = []

for fam in family:

    families.append(train[train["Parch"] == fam].Survived.value_counts())

df = pd.DataFrame(families)

df.index = family

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
siblings = sorted(train.SibSp.unique())[1:]

sibs = []

for sib in siblings:

    sibs.append(combined[combined["SibSp"] == sib].Pclass.value_counts())

df = pd.DataFrame(sibs)

df.index = siblings

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
family = sorted(train.Parch.unique())[1:]

families = []

for fam in family:

    families.append(combined[combined["Parch"] == fam].Pclass.value_counts())

df = pd.DataFrame(families)

df.index = family

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
family = sorted(train.Parch.unique())

families = []

for fam in family:

    families.append(combined[combined["Parch"] == fam].SibSp.value_counts())

df = pd.DataFrame(families)

df.index = family

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()

family = sorted(train.Parch.unique())[1:]

families = []

for fam in family:

    families.append(combined[combined["Parch"] == fam].SibSp.value_counts())

df = pd.DataFrame(families)

df.index = family

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
single = train[(train["Parch"] + train["SibSp"]) == 0].Survived.value_counts()

small_fam = train[((train["Parch"] + train["SibSp"]) > 0) & ((train["Parch"] + train["SibSp"]) <= 4)].Survived.value_counts()

big_fam = train[(train["Parch"] + train["SibSp"]) > 4].Survived.value_counts()

df = pd.DataFrame([single, small_fam, big_fam])

df.index = ["Single", "Small family", "Big family"]

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
single = combined[(combined["Parch"] + train["SibSp"]) == 0].Pclass.value_counts()

small_fam = combined[((combined["Parch"] + train["SibSp"]) > 0) & ((train["Parch"] + train["SibSp"]) <= 4)].Pclass.value_counts()

big_fam = combined[(combined["Parch"] + train["SibSp"]) > 4].Pclass.value_counts()

df = pd.DataFrame([single, small_fam, big_fam])

df.index = ["Single", "Small family", "Big family"]

df.plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()
title_df = train[["Name", "Survived"]].copy()

title_df["Title"] = title_df.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()

title_df.Title.value_counts()
titles = {"Mrs": "Miss",

          "Col": "military",

          "Ms": "Miss",

          "Mlle": "Miss",

          "Major": "military",

          "Sir": "royalty",

          "Dona": "royalty",

          "the Countess": "royalty",

          "Lady": "royalty",

          "Don": "royalty",

          "Capt": "royalty",

          "Jonkheer": "royalty",

          "Mme": "Miss"}



title_df.Title = title_df.Title.map(lambda x: titles[x] if x in titles.keys() else x)
title_df.head()
all_titles = list(title_df.Title.unique())

title_survive = []

for title in all_titles:

    title_survive.append(title_df[title_df["Title"] == title].Survived.value_counts())

df = pd.DataFrame(title_survive)

df.index = all_titles

df.columns = ["Dead", "Survived"]

df[["Survived", "Dead"]].plot(kind="bar", stacked=True)

plt.tight_layout()

plt.show()

df
combined.count()
combined.Age.describe()
fare_median = combined.Fare.median()

embarked_median = combined.Embarked.value_counts().index[0]

cabin_unkown = "U"



def ImputeAge(df):

    random.seed(1337) # reproducible

    genders = df.Sex.value_counts().index

    classes = sorted(combined.Pclass.value_counts().index)

    # generate dict

    genders_dict = {}

    classes_dict = {}

    for i in range(len(genders)):

        genders_dict[genders[i]] = i

    for i in range(len(classes)):

        classes_dict[classes[i]] = i

    # impute

    imputer = []

    for gender in genders:

        imputer.append([])

        for cls in classes:

            group = combined[(combined.Sex == gender) & 

                             (combined.Pclass == cls)]

            imputer[-1].append(gaussian_kde(group.Age.dropna()))

    max_age = df.Age.max()

    for index, row in df.iterrows():

        if np.isnan(row.Age):

            generator = imputer[genders_dict[row.Sex]][classes_dict[row.Pclass]]

            (x, y) = (random.uniform(0, max_age), random.random())

            while y > generator(x):

                (x, y) = (random.uniform(0, max_age), random.random())

            df.Age[index] = x

            

def FamilyState(row):

    if (row.Parch + row.SibSp) == 0:

        return "Single"

    elif (row.Parch + row.SibSp) <= 4:

        return "Small"

    else:

        return "Big"

    

def Title(df):

    titles = {"Mrs": "Miss",

              "Col": "military",

              "Ms": "Miss",

              "Mlle": "Miss",

              "Major": "military",

              "Sir": "royalty",

              "Dona": "royalty",

              "the Countess": "royalty",

              "Lady": "royalty",

              "Don": "royalty",

              "Capt": "royalty",

              "Jonkheer": "royalty",

              "Mme": "Miss"}

    df["Title"] = df.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()

    df["Title"] = df.Title.map(lambda x: titles[x] if x in titles.keys() else x)
ImputeAge(combined)

combined.Fare.fillna(fare_median, inplace=True)

combined.Embarked.fillna(embarked_median, inplace=True)

combined.Cabin = combined.Cabin.str[0]

combined.Cabin.fillna(cabin_unkown, inplace=True)

Title(combined)

combined["Family"] = combined.apply(lambda row: FamilyState(row), axis=1)

combined.drop(["Name", "Ticket", "Parch", "SibSp"], axis=1, inplace=True)

combined = pd.get_dummies(combined, columns=["Sex"], drop_first=True)

combined = pd.get_dummies(combined, columns=["Cabin", "Embarked", "Pclass", "Family", "Title"])

combined.head()
train = pd.read_csv("../input/train.csv")

train_target = train.Survived.copy()

train = combined[combined.PassengerId.isin(train.PassengerId)].copy()

test = pd.read_csv("../input/test.csv")

test = combined[combined.PassengerId.isin(test.PassengerId)].copy()

test_ids = test.PassengerId.copy()

train.drop(["PassengerId"], axis=1, inplace=True)

test.drop(["PassengerId"], axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel



params= {"n_estimators": 100,

         "criterion": "gini",

         "bootstrap": True,

         "max_features": "sqrt"}



clf = RandomForestClassifier()

clf.set_params(**params)

clf.fit(train, train_target)



features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))

plt.show()





model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

print("Reduced features to {0:d}".format(train_reduced.shape[1]))

test_reduced = model.transform(test)
# link: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")



clf.set_params(**params)

# Set: n_estimators, criterion, max_features, bootsrat

# rest was found using grid search

if False:

    parameters= {"max_depth": [1, 3, 5, 10],

                 "min_samples_split": [2, 5, 10],

                 "min_samples_leaf": [1, 3, 5, 10]}



    grid_search = GridSearchCV(clf, param_grid=parameters)

    grid_search.fit(train_reduced, train_target)

    report(grid_search.cv_results_, n_top=5)
parameters = {"n_estimators": 100,

              "criterion": "gini",

              "bootstrap": False,

              "max_features": "sqrt",

              "max_depth": 10,

              "min_samples_split": 2,

              "min_samples_leaf": 3}



clf = RandomForestClassifier()

clf.set_params(**parameters)

clf.fit(train_reduced, train_target)

print("Training score = {0:.3f}".format(clf.score(train_reduced, train_target)))



predictions = clf.predict(test_reduced)
predictions_series = pd.Series(predictions, name="Survived")

result_df = pd.concat([test_ids.reset_index(drop=True), predictions_series], axis=1)

result_df.head()

#result_df.to_csv("../input/predictions.csv", header=True, index=False)