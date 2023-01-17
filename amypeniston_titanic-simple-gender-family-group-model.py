import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV, cross_validate

from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import SVC

import xgboost as xgb



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



SEED = 42

NFOLDS = 10



train_ = pd.read_csv("../input/titanic/train.csv")

test_ = pd.read_csv("../input/titanic/test.csv")
test = test_.copy()

train = train_.copy()

test_train = pd.concat([test, train], sort=False)

train.head()
def extract_title(x):

    return x.split(', ')[1].split(". ")[0].strip()
for dataset in [train, test, test_train]:

    dataset["Title"] = dataset["Name"].apply(extract_title)
def extract_last_name(x):

    return x.split(",")[0].strip()
for dataset in [train, test, test_train]:

    dataset["LastName"] = dataset["Name"].apply(extract_last_name)
for dataset in [train, test, test_train]:

    dataset["Age"] = dataset["Age"].fillna(99)
for dataset in [train, test, test_train]:

    dataset["Fare"] = dataset["Fare"].fillna(0)
for dataset in [train, test, test_train]:

    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])
for dataset in [train, test, test_train]:

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

    

sns.catplot(x="FamilySize", y="Survived", data=train, kind="bar")

plt.show()
for dataset in [train, test, test_train]:

    dataset["IsSolo"] = dataset["FamilySize"] == 1



sns.catplot(x="IsSolo", y="Survived", data=train, kind="bar")

plt.show()
for dataset in [train, test, test_train]:

    dataset["FamilyGroup"] = dataset["Pclass"].astype(str) + " - " + dataset["Ticket"].str[:-1] + " - " + dataset["Embarked"] + " - " + dataset["Fare"].astype(str)

    dataset["FamilyGroupOld"] = dataset["LastName"] + " - " + dataset["Ticket"].str[:-1]
train.loc[train["LastName"] == "Andersson"]
train.loc[train["FamilyGroup"] == "3 - 34708 - S - 31.275"]
train.loc[train["LastName"] == "Vander Planke"]
train[train["Title"] == "Master"].head()
masters = train.loc[(train["Title"] == "Master") & (train["Age"] != 99)]

sns.distplot(masters["Age"].dropna(), bins=7)

plt.ylabel("Density")

plt.show()
masters["Age"].describe()
boys_without_master = train.loc[(train["Age"] < 18) & (train["Sex"] == "male") & (train["Title"] != "Master")]

boys_without_master
len(boys_without_master)
test_train_group_count = test_train.loc[(test_train["Sex"] == "female") | (test_train["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]].sort_index()

test_train_group_count.columns = ["Train + Test Count"]



train_group_count = pd.merge(train.loc[(train["Sex"] == "female") | (train["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]], 

                             train.loc[(train["Sex"] == "female") | (train["Title"] == "Master")].groupby("FamilyGroup").sum()[["Survived"]], how="inner", on="FamilyGroup")

train_group_count.columns = ["Train Count", "Survived"]



test_group_count = test.loc[(test["Sex"] == "female") | (test["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]].sort_index()

test_group_count.columns = ["Test Count"]
groups = pd.merge(pd.merge(test_train_group_count, train_group_count, how="left", on="FamilyGroup"), test_group_count, how="left", on="FamilyGroup")

groups["Train Survival Rate"] = groups["Survived"] / groups["Train Count"]

groups = groups.reset_index()

groups.head()
temp = test_train.sort_values(by="Ticket")

fg_counts1 = temp.groupby("FamilyGroup").count().iloc[:,0]

fg_counts2 = temp.groupby("FamilyGroupOld").count().iloc[:,0]



fg_comparison = pd.merge(pd.merge(temp[["PassengerId", "FamilyGroup", "FamilyGroupOld"]], fg_counts1, on="FamilyGroup", how="left"), fg_counts2, on="FamilyGroupOld", how="left")

fg_comparison.columns = ["PassengerId", "FamilyGroup", "FamilyGroupOld", "CountFamilyGroup", "CountFamilyGroupOld"]

fg_comparison["CountFamilyGroup"] = fg_comparison["CountFamilyGroup"].astype(int)

fg_comparison = fg_comparison.sort_values(by="FamilyGroup")

fg_comparison[fg_comparison["CountFamilyGroup"] != fg_comparison["CountFamilyGroupOld"]].head()
train[train["FamilyGroup"] == "1 - 11081 - C - 75.25"]
train[train["FamilyGroup"] == "1 - 11378 - S - 151.55"]
familygroups = groups[groups["Train + Test Count"] > 1][["FamilyGroup"]]

familygroups.head()
families = groups[groups["FamilyGroup"].isin(familygroups["FamilyGroup"])]

families.head()
test_males_xmasters_ids = test.loc[(test["Sex"] == "male") & (test["Title"] != "Master")]["PassengerId"]

test_females_masters_ids = test.loc[(test["Sex"] == "female") | (test["Title"] == "Master")]["PassengerId"]
test_males_xmasters_preds = pd.DataFrame({"PassengerId": test_males_xmasters_ids, "Survived": np.zeros(len(test_males_xmasters_ids), dtype=int)})

test_males_xmasters_preds.head()
len(test_males_xmasters_preds)
test_females_masters = test.loc[test["PassengerId"].isin(test_females_masters_ids)]

test_females_masters_rates = pd.merge(test_females_masters, groups, how="left", on="FamilyGroup")[["PassengerId", "FamilyGroup", "Sex", "Age", "Train + Test Count", "Train Count", "Survived", "Train Survival Rate"]]

test_females_masters_rates["Train Survival Rate"].fillna(-1, inplace=True)

test_females_masters_rates.head()
test_females_masters_rates["Prediction"] = np.zeros(len(test_females_masters_rates), dtype=int)

test_females_masters_rates.loc[(test_females_masters_rates["Sex"] == "female"), "Prediction"] = 1

test_females_masters_rates.loc[(test_females_masters_rates["Train Survival Rate"] >= 0.5), "Prediction"] = 1

test_females_masters_rates.loc[(test_females_masters_rates["Train Survival Rate"] < 0.5) & (test_females_masters_rates["Train Survival Rate"] != -1), "Prediction"] = 0

test_females_masters_rates.head()
test_females_masters_preds = test_females_masters_rates[["PassengerId", "Prediction"]]

test_females_masters_preds.columns = ["PassengerId", "Survived"]

output = pd.concat([test_males_xmasters_preds, test_females_masters_preds]).sort_values(by="PassengerId")

output.to_csv("submission.csv", index=False)

output.head()
np.sum(output["Survived"])
len(test[test["Sex"] == "female"])