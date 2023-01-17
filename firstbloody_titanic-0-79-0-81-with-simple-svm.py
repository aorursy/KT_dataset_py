# data processing and visualization

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# algorithm

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC

# training训练

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")
train.describe(include="all")
sns.barplot(x="Sex", y="Survived", data=train)

print("女性生存率:", train["Survived"][train["Sex"] == "female"].value_counts(normalize=True)[1])

print("男性生存率:", train["Survived"][train["Sex"] == "male"].value_counts(normalize=True)[1])
train["Family_size"] = train["SibSp"] + train["Parch"]

test["Family_size"] = test["SibSp"] + test["Parch"]
train["Fname"] = train.Name.apply(lambda x: x.split(",")[0])

test["Fname"] = test.Name.apply(lambda x: x.split(",")[0])
train[train.Fname == "Vander Planke"]
test[test.Fname == "Vander Planke"]
train[train.Fname == "Allison"]
train[(train.Fname == "Hoyt") & (train.Family_size > 0)]
train[(train.Fname == "Moubarek") & (train.Family_size > 0)]
dead_train = train[train["Survived"] == 0]

fname_ticket = dead_train[(dead_train["Sex"] == "female") & (dead_train["Family_size"] >= 1)][["Fname", "Ticket"]]

train["dead_family"] = np.where(train["Fname"].isin(fname_ticket["Fname"]) & train["Ticket"].isin(fname_ticket["Ticket"]) & ((train["Age"] >=1) | train.Age.isnull()), 1, 0)

test["dead_family"] = np.where(test["Fname"].isin(fname_ticket["Fname"]) & test["Ticket"].isin(fname_ticket["Ticket"]) & ((test["Age"] >=1) | test.Age.isnull()), 1, 0)
live_train = train[train["Survived"] == 1]

live_fname_ticket = live_train[(live_train["Sex"] == "male") & (live_train["Family_size"] >= 1) & ((live_train["Age"] >= 18) | (live_train["Age"].isnull()))][["Fname", "Ticket"]]

train["live_family"] = np.where(train["Fname"].isin(live_fname_ticket["Fname"]) & train["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)

test["live_family"] = np.where(test["Fname"].isin(live_fname_ticket["Fname"]) & test["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)
dead_man_fname_ticket = train[(train["Family_size"] >= 1) & (train["Sex"] == "male") & (train["Survived"] == 0) & (train["dead_family"] == 0)][["Fname", "Ticket"]]

train["deadfamily_man"] = np.where(train["Fname"].isin(dead_man_fname_ticket["Fname"]) & train["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (train.Sex == "male"), 1, 0)

train["deadfamily_woman"] = np.where(train["Fname"].isin(dead_man_fname_ticket["Fname"]) & train["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (train.Sex == "female"), 1, 0)

test["deadfamily_man"] = np.where(test["Fname"].isin(dead_man_fname_ticket["Fname"]) & test["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (test.Sex == "male"), 1, 0)

test["deadfamily_woman"] = np.where(test["Fname"].isin(dead_man_fname_ticket["Fname"]) & test["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (test.Sex == "female"), 1, 0)

train.loc[(train["dead_family"] == 0) & (train["live_family"] == 0) & (train["deadfamily_man"] == 0) & (train["deadfamily_woman"] == 0) & (train["Family_size"] >= 1) & (train["Sex"] == "male"), "deadfamily_man"] = 1

train.loc[(train["dead_family"] == 0) & (train["live_family"] == 0) & (train["deadfamily_man"] == 0) & (train["deadfamily_woman"] == 0) & (train["Family_size"] >= 1) & (train["Sex"] == "female"), "deadfamily_woman"] = 1

test.loc[(test["dead_family"] == 0) & (test["live_family"] == 0) & (test["deadfamily_man"] == 0) & (test["deadfamily_woman"] == 0) & (test["Family_size"] >= 1) & (test["Sex"] == "male"), "deadfamily_man"] = 1

test.loc[(test["dead_family"] == 0) & (test["live_family"] == 0) & (test["deadfamily_man"] == 0) & (test["deadfamily_woman"] == 0) & (test["Family_size"] >= 1) & (test["Sex"] == "female"), "deadfamily_woman"] = 1
grp_tk = train.drop(["Survived"], axis=1).append(test).groupby(["Ticket"])

tickets = []

for grp, grp_train in grp_tk:

    ticket_flag = True

    if len(grp_train) != 1:

        for i in range(len(grp_train) - 1):

            if grp_train.iloc[i]["Fname"] != grp_train.iloc[i+1]["Fname"]:

                ticket_flag = False

    if ticket_flag == False:

        tickets.append(grp)

train.loc[(train.Ticket.isin(tickets)) & (train.Family_size == 0) & (train.Sex == "male"), "deadfamily_man"] = 1

train.loc[(train.Ticket.isin(tickets)) & (train.Family_size == 0) & (train.Sex == "female"), "deadfamily_woman"] = 1

test.loc[(test.Ticket.isin(tickets)) & (test.Family_size == 0) & (test.Sex == "male"), "deadfamily_man"] = 1

test.loc[(test.Ticket.isin(tickets)) & (test.Family_size == 0) & (test.Sex == "female"), "deadfamily_woman"] = 1
test = test.fillna({"Fare": test[test["Pclass"] == 3]["Fare"].mean()})
train = train.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname"], axis=1)

test = test.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname"], axis=1)
train_dummies_sex = pd.get_dummies(train["Sex"])

test_dummies_sex = pd.get_dummies(test["Sex"])

train = pd.concat([train, train_dummies_sex], axis=1)

test = pd.concat([test, test_dummies_sex], axis=1)

train = train.drop(["Sex"], axis=1)

test = test.drop(["Sex"], axis=1)
train_name = train.Name.str.extract("([a-zA-Z]+)\.")

test_name = test.Name.str.extract("([a-zA-Z]+)\.")

train_name["Title"] = train.Name.str.extract("([a-zA-Z]+)\.")

test_name["Title"] = test.Name.str.extract("([a-zA-Z]+)\.")

 

train_name = train_name.drop([0], axis=1)

test_name = test_name.drop([0], axis=1)

 

train_name["Title"] = train_name["Title"].replace(["Mlle", "Ms"], "Miss")

train_name["Title"] = train_name["Title"].replace(["Mme"], "Mrs")

train_name["Title"] = train_name["Title"].replace(["Countess", "Sir", "Lady", "Don"], "Royal")

train_name["Title"] = train_name["Title"].replace(["Dr", "Rev", "Col", "Major", "Jonkheer", "Capt"], "Rare")

 

test_name["Title"] = test_name["Title"].replace(["Ms"], "Miss")

test_name["Title"] = test_name["Title"].replace(["Dona"], "Mrs")

test_name["Title"] = test_name["Title"].replace(["Dr", "Rev", "Col"], "Rare")

 

train_name["Title"] = train_name["Title"].replace(["Mr"], 1)

train_name["Title"] = train_name["Title"].replace(["Miss"], 2)

train_name["Title"] = train_name["Title"].replace(["Mrs"], 3)

train_name["Title"] = train_name["Title"].replace(["Master"], 4)

train_name["Title"] = train_name["Title"].replace(["Royal"], 5)

train_name["Title"] = train_name["Title"].replace(["Rare"], 6)

 

test_name["Title"] = test_name["Title"].replace(["Mr"], 1)

test_name["Title"] = test_name["Title"].replace(["Miss"], 2)

test_name["Title"] = test_name["Title"].replace(["Mrs"], 3)

test_name["Title"] = test_name["Title"].replace(["Master"], 4)

test_name["Title"] = test_name["Title"].replace(["Rare"], 6)

 

train["Title"] = train_name["Title"]

test["Title"] = test_name["Title"]

 

train = train.drop(["Name"], axis=1)

test = test.drop(["Name"], axis=1)
age_train = pd.concat([train.drop(["Survived"], axis=1), test], axis=0)

age_train = age_train[age_train["Age"].notnull()]

 

age_label = age_train["Age"]

age_train = age_train.drop(["Age"], axis=1)

 

RFR = RandomForestRegressor(max_depth=16, n_estimators=16)

RFR.fit(age_train, age_label)

 

train.loc[train.Age.isnull(), ["Age"]] = RFR.predict(train[train.Age.isnull()].drop(["Age", "Survived"], axis=1))

test.loc[test.Age.isnull(), ["Age"]] = RFR.predict(test[test.Age.isnull()].drop(["Age"], axis=1))
train = train.drop(["SibSp", "Parch"], axis=1)

test = test.drop(["SibSp", "Parch"], axis=1)
train.loc[train["Age"] <= 15, "AgeBin"] = 0

train.loc[(train["Age"] > 15) & (train["Age"] <= 30), "AgeBin"] = 1

train.loc[(train["Age"] > 30) & (train["Age"] <= 49), "AgeBin"] = 2

train.loc[(train["Age"] > 49) & (train["Age"] < 80), "AgeBin"] = 3

train.loc[train["Age"] >= 80, "AgeBin"] = 4

test.loc[test["Age"] <= 15, "AgeBin"] = 0

test.loc[(test["Age"] > 15) & (test["Age"] <= 30), "AgeBin"] = 1

test.loc[(test["Age"] > 30) & (test["Age"] <= 49), "AgeBin"] = 2

test.loc[(test["Age"] > 49) & (test["Age"] < 80), "AgeBin"] = 3

test.loc[test["Age"] >= 80, "AgeBin"] = 4
pd.qcut(train.drop(["Survived"], axis=1).append(test)["Fare"], 5).head(10)
train.loc[train["Fare"] <= 7.854, "FareBin"] = 0

train.loc[(train["Fare"] > 7.854) & (train["Fare"] <= 10.5), "FareBin"] = 1

train.loc[(train["Fare"] > 10.5) & (train["Fare"] <= 21.558), "FareBin"] = 2

train.loc[(train["Fare"] > 21.558) & (train["Fare"] <= 41.579), "FareBin"] = 3

train.loc[train["Fare"] > 41.579, "FareBin"] = 4

test.loc[test["Fare"] <= 7.854, "FareBin"] = 0

test.loc[(test["Fare"] > 7.854) & (test["Fare"] <= 10.5), "FareBin"] = 1

test.loc[(test["Fare"] > 10.5) & (test["Fare"] <= 21.558), "FareBin"] = 2

test.loc[(test["Fare"] > 21.558) & (test["Fare"] <= 41.579), "FareBin"] = 3

test.loc[test["Fare"] > 41.579, "FareBin"] = 4
train = train.drop(["Age", "Fare"], axis=1)

test = test.drop(["Age", "Fare"], axis=1)
train.head(10)
y = train["Survived"]

train_x, val_x, train_y, val_y = train_test_split(train.drop(["Survived"], axis=1), y, test_size=0.22, random_state=0)

clf = SVC(C=1, probability=True)

clf.fit(train_x, train_y)

clf.score(val_x, val_y)
svc_grid = GridSearchCV(SVC(), {"C": [i for i in range(1, 101)]}, cv=4)

svc_grid.fit(train.drop(["Survived"], axis=1), y)

svc_grid.best_params_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

 

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

 

    plt.legend(loc="best")

    return plt
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(SVC(C=10), "C=10", train.drop(["Survived"], axis=1), y, cv=cv)
clf = SVC(C=10, probability=True)

clf.fit(train_x, train_y)

gender_submission["Survived"] = clf.predict(test)

gender_submission.to_csv("1.csv", index=False)