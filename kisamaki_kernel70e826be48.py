import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import StratifiedKFold, cross_validate 

import optuna



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

data = pd.concat([train, test])



print(data.info())

print(train["Survived"].mean())
train["Family_size"] = train["SibSp"] + train["Parch"]

train["Family_size_bin"] = "big"

train.loc[train["Family_size"]==0, "Family_size_bin"] = "small"

train.loc[(train["Family_size"]>=1) & (train["Family_size"]<=3), "Family_size_bin"] = "midiam"

sns.countplot(train["Family_size_bin"], hue=train["Survived"])

print(train["Survived"].groupby(train["Family_size_bin"]).agg(["mean", "count"]))
train["Cabin_top"] = train["Cabin"].map(lambda x: str(x)[0])

train["Cabin_top"].replace(["F", "G", "T"], "R", inplace=True)

train["Cabin_top"].replace(["A", "B", "C", "D", "E", "R"], "E", inplace=True)

sns.countplot(train["Cabin_top"], hue=train["Survived"])

train["Survived"].groupby(train["Cabin_top"]).agg(["mean", "count"])
train["Middle"] = train["Name"].map(lambda x: x.split(", ")[1].split(".")[0])

train["Middle"].replace(["Lady", "Ms", "Mlle", "Mme", "the Countess"], 'Miss',inplace=True)

train["Middle"].replace([], 'Mr',inplace=True)

train["Middle"].replace(["Dr","Col", "Rev", "Capt", "Don", "Jonkheer", "Major", "Sir"],"Rare", inplace=True)

print(train["Survived"].groupby(train["Middle"]).agg(["mean", "count"]))

sns.countplot(train["Middle"], hue=train["Survived"])



test["Middle"] = test["Name"].map(lambda x: x.split(", ")[1].split(".")[0])

test["Middle"].replace(["Lady", "Ms", "Mlle", "Mme", "the Countess"], 'Miss',inplace=True)

test["Middle"].replace([], 'Mr',inplace=True)

test["Middle"].replace(["Dr","Col", "Rev", "Capt", "Don", "Jonkheer", "Major", "Sir", "Dona"],"Rare", inplace=True)

test["Middle"].value_counts()
print(train["Age"].mean())

print(train.corr()["Age"])

sns.catplot(x = "Middle", y = 'Age', data = train, kind = "box")

sns.catplot(x = "Pclass", y = 'Age', data = train, kind = "box")

sns.catplot(x = "Embarked", y = 'Age', data = train, kind = "box")

sns.catplot(x = "Survived", y = "Age", data = train, kind = "box")
data["Middle"] = data["Name"].map(lambda x: x.split(", ")[1].split(".")[0])

data["Middle"].replace(["Lady", "Ms", "Mlle", "Mme", "the Countess"], 'Miss',inplace=True)

data["Middle"].replace([], 'Mr',inplace=True)

data["Middle"].replace(["Dr","Col", "Rev", "Capt", "Don", "Jonkheer", "Major", "Sir"],"Rare", inplace=True)

age_pre = data[["Middle", "Pclass", "SibSp", "Parch"]]

age_pre = pd.get_dummies(age_pre)



x_train_age = age_pre[data["Age"].notnull()]

x_unknown_age = age_pre[data["Age"].isnull()]

y_train_age = data["Age"][data["Age"].notnull()]



rfc_age = RandomForestRegressor(random_state=0)

rfc_age = rfc_age.fit(x_train_age, y_train_age)

predict_age = rfc_age.predict(x_unknown_age)

data.loc[(data["Age"].isnull()), ["Age"]] = predict_age

train["Age"] = train["Age"][:len(train)]

sns.catplot(x = "Middle", y = 'Age', data = train, kind = "box")
train[train["Fare"].isna()]

print(test[test["Fare"].isna()])

sns.catplot(x = "Pclass", y = "Fare", data = train, kind = "box")
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

data = pd.concat([train, test])



train["Embarked"].fillna("S", inplace=True)



train["Family_size"] = train["SibSp"] + train["Parch"]

train["Family_size_bin"] = "big"

train.loc[train["Family_size"]==0, "Family_size_bin"] = "small"

train.loc[(train["Family_size"]>=1) & (train["Family_size"]<=3), "Family_size_bin"] = "midiam"



train["alone"] = 0

train.loc[train["Family_size"] != 0, "alone"] = 1



train["Cabin_top"] = train["Cabin"].map(lambda x: str(x)[0])

train["Cabin_top"].replace(["F", "G", "T"], "R", inplace=True)

train["Cabin_top"].replace(["A", "B", "C", "D", "E", "R"], "E", inplace=True)



train["Middle"] = train["Name"].map(lambda x: x.split(", ")[1].split(".")[0])

train["Middle"].replace(["Lady", "Ms", "Mlle", "Mme", "the Countess"], 'Miss',inplace=True)

train["Middle"].replace([], 'Mr',inplace=True)

train["Middle"].replace(["Dr","Col", "Rev", "Capt", "Don", "Jonkheer", "Major", "Sir"],"Rare", inplace=True)



predict_age = rfc_age.predict(x_unknown_age)

data.loc[(data["Age"].isnull()), ["Age"]] = predict_age

train["Age"] = data["Age"][:len(train)]



train = pd.get_dummies(train, columns=["Family_size_bin"])



delete_columns = ["Name", "Family_size", "SibSp", "Parch", "PassengerId", "Cabin_top", "Middle"]

train = train.drop(delete_columns, axis=1)

plt.figure(figsize=(10, 8))

sns.heatmap(train.corr(), annot=True)

train.info()
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



data = pd.concat([train, test])



data["Sex"].replace(["male", "female"], [0, 1], inplace=True)



predict_age = rfc_age.predict(x_unknown_age)

data.loc[(data["Age"].isnull()), ["Age"]] = predict_age



data["Fare"].fillna(0, inplace=True)



data["Embarked"].fillna("S", inplace=True)

data["Embarked"].replace(["S", "Q", "C"], [0, 1, 2], inplace=True)



data["Family_size"] = data["SibSp"] + data["Parch"]

data["Family_size_bin"] = "big"

data.loc[data["Family_size"]==0, "Family_size_bin"] = "small"

data.loc[(data["Family_size"]>=1) & (data["Family_size"]<=3), "Family_size_bin"] = "midiam"



data["alone"] = 0

data.loc[data["Family_size"] != 0, "alone"] = 1



data["Cabin_top"] = data["Cabin"].map(lambda x: str(x)[0])

data["Cabin_top"].replace(["F", "G", "T"], "R", inplace=True)

data["Cabin_top"].replace(["A", "B", "C", "D", "E", "R"], "E", inplace=True)



data["Middle"] = data["Name"].map(lambda x: x.split(", ")[1].split(".")[0])

data["Middle"].replace(["Lady", "Ms", "Mlle", "Mme", "the Countess"], 'Miss',inplace=True)

data["Middle"].replace([], 'Mr',inplace=True)

data["Middle"].replace(["Dr","Col", "Rev", "Capt", "Don", "Jonkheer", "Major", "Sir", "Dona"],"Rare", inplace=True)



#data = pd.get_dummies(data, columns=["Family_size_bin"])



delete_columns = ["Family_size_bin", "PassengerId", "Name", "Ticket", "Cabin", "Family_size", "SibSp", "Parch", "Cabin_top", "Middle"]

data = data.drop(delete_columns, axis=1)



train = data[:len(train)]

test = data[len(train):]



x_train = train.drop("Survived", axis=1)

x_test = test.drop("Survived", axis=1)

y_train = train["Survived"]



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rfc = RandomForestClassifier(n_estimators=60,

                             max_depth=17, 

                             min_samples_leaf=1, 

                             min_samples_split=14,

                             criterion="entropy",

                             random_state=0)

scores = cross_validate(rfc, x_train, y_train, cv=skf)

print(scores['test_score'].mean())

rfc.fit(x_train, y_train)



y_test = rfc.predict(x_test)

test = pd.read_csv("../input/titanic/test.csv")

sub = pd.concat([

    pd.DataFrame(test["PassengerId"], columns=["PassengerId"]),

    pd.DataFrame(y_test, columns=['Survived'])

], axis=1)

sub["Survived"] = sub["Survived"].map(lambda x: int(x)) 

sub.to_csv("sub.csv", index=False)
def objective(trial):

    

    param_grid_rfc = {

        "n_estimators": int(trial.suggest_discrete_uniform("n_estimators", 40, 70, 10)),

        "max_depth": int(trial.suggest_discrete_uniform("max_depth", 15, 25, 1)),

        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),

        'min_samples_split': trial.suggest_int("min_samples_split", 12, 14),

        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),

        "random_state": 0

    }



    model = RandomForestClassifier(**param_grid_rfc)

    

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)

    return scores['test_score'].mean()



#study = optuna.create_study(direction='maximize')

#study.optimize(objective, n_trials=100)

#print(study.best_params)

#print(study.best_value)