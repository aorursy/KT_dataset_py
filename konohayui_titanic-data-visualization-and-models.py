import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
train.head(10)
train.describe()
sns.countplot("Survived", data = train)
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train["Sex"].value_counts().plot.bar(color = "skyblue", ax = ax[0])

ax[0].set_title("Number Of Passengers By Sex")

ax[0].set_ylabel("Population")

sns.countplot("Sex", hue = "Survived", data = train, ax = ax[1])

ax[1].set_title("Sex: Survived vs Dead")

plt.show()
sns.factorplot("Pclass", "Survived", hue = "Sex", data = train)

plt.show()
pd.crosstab([train["Sex"], train["Survived"]], train["Pclass"], 

            margins = True).style.background_gradient(cmap = "summer_r")
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train["Pclass"].value_counts().plot.bar(color = "skyblue", ax = ax[0])

ax[0].set_title("Number Of Passengers By Pclass")

ax[0].set_ylabel("Population")

sns.countplot("Pclass", hue = "Survived", data = train, ax = ax[1])

ax[1].set_title("Pclass: Survived vs Dead")

plt.show()
sns.barplot(x = "Sex", y = "Survived", hue = "Pclass", data = train)

plt.show()
def get_title(data_frame):

    # Get names from data frame

    name_data = data_frame["Name"]

    

    # Obtain titles

    data_frame["Title"] = [name.split(", ", 1)[1].split(".", 1)[0] for name in name_data]

    

    # Find all titles

    titles = []

    for title in data_frame["Title"]:

        if title not in titles:

            titles.append(title)        

    

    return data_frame, titles



train, titles = get_title(train)

print(titles)
pd.crosstab(train["Title"], train["Sex"]).T.style.background_gradient(cmap = "summer_r")
def title2int(data):

    data["Title"].replace(["Major", "Capt", "Sir", "Dr", "Don", "Mlle", "Mme", "Ms", "Dona", "Lady", "the Countess", "Jonkheer", "Col", "Rev"],

                           ["Mr", "Mr", "Mr", "Mr", "Mr", "Miss", "Miss", "Miss", "Mrs", "Mrs", "Mrs", "Other", "Other", "Other"], inplace = True)

    data["Title"].replace(["Mr", "Miss", "Mrs", "Master", "Other"], [0, 1, 2, 3, 4], inplace = True)

    return data



train = title2int(train)
train.groupby("Title")["Age"].mean()
sns.countplot("Title", hue = "Survived", data = train)

plt.show()
pd.crosstab([train["Title"], train["Survived"]], train["Pclass"], margins = True).style.background_gradient(cmap = "summer_r")
sns.boxplot(train["Fare"])

plt.show()
fig, ax = plt.subplots(1, 3, figsize = (15, 5))

sns.boxplot(train[train["Pclass"] == 1]["Fare"], ax = ax[0])

ax[0].set_title("Fares in Pclass 1")

sns.boxplot(train[train["Pclass"] == 2]["Fare"], ax = ax[1])

ax[1].set_title("Fares in Pclass 2")

sns.boxplot(train[train["Pclass"] == 3]["Fare"], ax = ax[2])

ax[2].set_title("Fares in Pclass 3")

plt.show()
train.groupby("Pclass")["Fare"].mean()
train.groupby("Pclass")["Fare"].median()
def fareG2int(data):

    data["Fare_group"] = "NaN"

    data.loc[data["Fare"] < 10, "Fare_group"] = 2 # class three

    data.loc[(data["Fare"] >= 10) & (data["Fare"] < 65), "Fare_group"] = 2 # class one and class two

    data.loc[data["Fare"] >= 65, "Fare_group"] = 1 # the patricians

    return data



train = fareG2int(train)
sns.countplot("Fare_group", hue = "Survived", data = train)

plt.show()
train["Embarked"] = train["Embarked"].fillna("S")
pd.crosstab([train["Embarked"], train["Survived"]], train["Sex"], margins = True).style.background_gradient(cmap = "summer_r")
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train["Embarked"].value_counts().plot.bar(color = "skyblue", ax = ax[0])

ax[0].set_title("Number Of Passengers By Embarked")

ax[0].set_ylabel("Number")

sns.countplot("Embarked", hue = "Survived", data = train, ax = ax[1])

ax[1].set_title("Embarked: Survived vs Unsurvived")

plt.show()
pd.crosstab([train["Embarked"], train["Survived"]], train["Pclass"], margins = True).style.background_gradient(cmap = "summer_r")
sns.barplot(x = "Embarked", y = "Survived", hue = "Pclass", data = train)

plt.show()
train["Embarked"].replace(["S", "Q", "C"], [0, 1, 2], inplace = True)
pd.crosstab([train["Pclass"], train["Survived"]], train["SibSp"], margins = True).style.background_gradient(cmap = "summer_r")
pd.crosstab([train["Pclass"], train["Survived"]], train["Parch"], margins = True).style.background_gradient(cmap = "summer_r")
def Cabin_type(data):

    data.loc[data["Cabin"].notnull(), "Cabin"] = "Known"

    data.loc[data["Cabin"].isnull(), "Cabin"] = "Unknown"

    return data



train = Cabin_type(train)

sns.countplot("Cabin", hue = "Survived", data = train)

plt.show()
def cab2int(data):

    data.loc[data["Cabin"] == "Known", 'Cabin'] = 1

    data.loc[data["Cabin"] == "Unknown", 'Cabin'] = 0

    return data



train = cab2int(train)
train["Sex"].replace(["male", "female"], [0, 1], inplace = True)
from sklearn.ensemble import RandomForestRegressor



def assign_missing_ages(data_frame, features):

    age_data = data_frame[features]

    known_ages = age_data[age_data.Age.notnull()].as_matrix()

    unknown_ages = age_data[age_data.Age.isnull()].as_matrix()

    

    # Create target and eigenvalues for known ages

    target = known_ages[:, 0]

    eigen_val = known_ages[:, 1:]

    

    # apply random forest regressor

    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)

    rfr.fit(eigen_val, target)

    

    # predictions

    Age_predictions = rfr.predict(unknown_ages[:, 1::])

    data_frame.loc[(data_frame.Age.isnull()), "Age"] = Age_predictions

    

    return data_frame, rfr



age_features = ["Age", "Sex", "SibSp", "Parch", "Pclass"]

train, rfr = assign_missing_ages(train, age_features)

train["Age"] = train["Age"].astype(int)

train.head(10)
train.groupby("Title")["Age"].mean()
fig, ax = plt.subplots(1, 2, figsize = (18, 10))

train[train["Survived"] == 0]["Age"].plot.hist(ax = ax[0], bins = 20, edgecolor = "black", color = "blue")

ax[0].set_title("Unsurvived")

domain_1 = list(range(0, 85, 5))

ax[0].set_xticks(domain_1)

train[train["Survived"] == 1]["Age"].plot.hist(ax = ax[1], bins = 20, edgecolor = "black", color = "green")

ax[1].set_title("Survived")

domain_2 = list(range(0, 85, 5))

ax[1].set_xticks(domain_2)

plt.show()
def ageG2int(data):

    data["Age_group"] = "NaN"

    data.loc[data["Age"] <= 16, "Age_group"] = 0 # Child

    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age_group"] = 1 # young teen and teen adult

    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age_group"] = 3 # middle age

    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_group"] = 4 # young elderly

    data.loc[data["Age"] > 64, "Age_group"] = 5 # elderly

    return data



train = ageG2int(train)
sns.countplot("Age_group", hue = "Survived", data = train)

plt.show()
pd.crosstab([train["Age_group"], train["Survived"]], train["Pclass"], margins = True).style.background_gradient(cmap = "summer_r")
sns.barplot(x = "Age_group", y = "Survived", hue = "Pclass", data = train)

plt.show()
pd.crosstab([train["Age_group"], train["Survived"]], [train["Pclass"], train["Sex"]], margins = True).style.background_gradient(cmap = "summer_r")
def child2int(data):

    data["Child"] = "NaN"

    data.loc[data["Age"] <= 18, "Child"] = 0 # Child

    data.loc[data["Age"] > 18, "Child"] = 1 # Adult

    return data



train = child2int(train)
sns.countplot("Child", hue = "Survived", data = train, palette = "Greens_d")

plt.show()
train["FamSize"] = train["SibSp"] + train["Parch"] + 1
def famG2int(data):

    data["Fam_group"] = "NaN"

    data.loc[data["FamSize"] == 1, "Fam_group"] = 0 # Single

    data.loc[data["FamSize"] > 1, "Fam_group"] = 1 # Family

#     data.loc[data["FamSize"] == 2, "Fam_group"] = 1 # Couple

#     data.loc[(data["FamSize"] > 2) & (data["FamSize"] <= 4), "Fam_group"] = 2 # Medium Family

#     data.loc[data["FamSize"] > 4, "Fam_group"] = 3 # Big Family

    return data



train = famG2int(train)
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train["Fam_group"].value_counts().plot.bar(color = "skyblue", ax = ax[0])

ax[0].set_title("Number Of Passengers By Family Group")

ax[0].set_ylabel("Number")

sns.countplot("Fam_group", hue = "Survived", data = train, ax = ax[1])

ax[1].set_title("Family Group: Survived vs Unsurvived")

plt.show()
pd.crosstab([train["Fam_group"], train["Survived"]], train["Pclass"], margins = True).style.background_gradient(cmap = "summer_r")
train_one = train[:]
# Reordering columns

columns_titles = ["PassengerId", "Survived", "Pclass", "Title", "Sex", "Child", "Fam_group", "Fare", "Cabin", "Embarked"]

train_one = train_one[columns_titles]

train_one.head(10)


test["Embarked"].replace(["S", "Q", "C"], [0, 1, 2], inplace = True)

test["Fare"] = test["Fare"].fillna(test["Fare"].median())



test, test_titles = get_title(test)

test = title2int(test)

test["Sex"].replace(["male", "female"], [0, 1], inplace = True)

test = Cabin_type(test)

test = cab2int(test)

test = fareG2int(test)



temp_test = test[age_features]

test_unknown_ages = temp_test[test["Age"].isnull()].as_matrix()

test_Age_predictions = rfr.predict(test_unknown_ages[:, 1:])

test.loc[(test["Age"].isnull()), "Age"] = test_Age_predictions

test["Age"] = test["Age"].astype(int)



test = ageG2int(test)

test = child2int(test)

test["FamSize"] = test["SibSp"] + test["Parch"] + 1

test = famG2int(test)



test_one = test[:]

test_columns_titles = ["PassengerId", "Pclass", "Title", "Sex", "Child", "Fam_group", "Fare", "Cabin", "Embarked"]

test_one = test_one[test_columns_titles]

test_one.info()

test_one.head()
from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



def my_models(model, X_train, Y_train, X_test, Y_test):

    my_model = model.fit(X_train, Y_train)

    

    print(my_model.feature_importances_)

    print(my_model.score(X_train, Y_train))

    

    model_prediction = my_model.predict(X_test)

    acc = metrics.accuracy_score(model_prediction, Y_test)

    

    return acc, my_model
final_features = ["Pclass", "Title", "Sex", "Child", "Fam_group", "Fare", "Cabin", "Embarked"]

final_data = train_one[["Survived"] + final_features]

training, testing = train_test_split(final_data, test_size = 0.3, random_state = 0, stratify = final_data["Survived"])

X_train = training[training.columns[1:]]

Y_train = training[training.columns[:1]]

X_test = testing[testing.columns[1:]]

Y_test = testing[testing.columns[:1]]
# from sklearn.model_selection import GridSearchCV



# parameter_grid_rf = {"n_estimators": [100, 200, 300, 400, 500], 

#                      "max_depth": [7, 8, 9, 10], 

#                      "max_leaf_nodes": [7, 8, 9, 10],

#                      "min_samples_leaf": [2, 3, 4, 5]}



# parameter_grid_gb = {"learning_rate": [0.1, 0.01, 0.005], 

#                   "n_estimators": [100, 200, 300, 400, 500], 

#                   "max_depth": [7, 8, 9, 10], 

#                   "subsample": [1.0, 0.5], 

#                   "max_features": [1.0, 0.5],

#                   "random_state": [0, 1]}



# forest_model = RandomForestClassifier()

# gradboost_model = GradientBoostingClassifier()



# grid_search = GridSearchCV(forest_model, parameter_grid_rf, cv = 10, verbose = 3)

# X = training.values[:, 1:]

# Y = training.values[:, 0].astype("int")

# grid_search.fit(X, Y)



# print(grid_search.best_score_)

# print(grid_search.best_params_)
tree_model = tree.DecisionTreeClassifier(max_depth = 8, max_leaf_nodes = 7, min_samples_leaf = 10, random_state = 0)

forest_model = RandomForestClassifier(max_depth = 8, max_leaf_nodes = 9, n_estimators = 300, random_state = 0)

gradboost_model = GradientBoostingClassifier(learning_rate =  0.01, max_depth = 7,

                                             max_features = 1.0, n_estimators = 200, subsample = 1.0, random_state = 0)
tree_acc, my_tree = my_models(tree_model, X_train, Y_train, X_test, Y_test)    

print("The accuracy of Decision Tree is", tree_acc)



forest_acc, my_forest = my_models(forest_model, X_train, Y_train, X_test, Y_test)    

print("The accuracy of Random Forest is", forest_acc)



gradboost_acc, my_gradboost = my_models(gradboost_model, X_train, Y_train, X_test, Y_test)    

print("The accuracy of Gradient Boosting is", gradboost_acc)
final_test = test_one[final_features]



tree_prediction = my_tree.predict(final_test)

forest_prediction = my_forest.predict(final_test)

gradboost_prediction = my_gradboost.predict(final_test)



test_cp1 = test_one[:]

test_cp2 = test_one[:]

test_cp3 = test_one[:]



headers = ["PassengerId", "Survived"]



test_cp1["Survived"] = tree_prediction

tree_prediction = pd.DataFrame(test_cp1, columns = headers)

tree_prediction.to_csv("tree_prediction.csv", index = False)



test_cp2["Survived"] = forest_prediction

forest_prediction = pd.DataFrame(test_cp2, columns = headers)

forest_prediction.to_csv("forest_prediction.csv", index = False)



test_cp3["Survived"] = gradboost_prediction

gradboost_prediction = pd.DataFrame(test_cp3, columns = headers)

gradboost_prediction.to_csv("gradboost_prediction.csv", index = False)