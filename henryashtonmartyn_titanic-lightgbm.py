import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import KBinsDiscretizer
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
train["PassengerId"].value_counts(dropna=False)
train["Survived"].value_counts(dropna=False)
train["Name"].value_counts(dropna=False)
train["Sex"].value_counts(dropna=False)
train["SibSp"].value_counts(dropna=False)
train["Parch"].value_counts(dropna=False)
train["Cabin"].value_counts(dropna=False)
train["Embarked"].value_counts(dropna=False)
train[train["Embarked"].isna()]
train["Fare"].value_counts(dropna=False).plot(kind="bar", figsize=(25, 5))
train["Fare"].max()
train["Fare"].min()
train["Ticket"].value_counts(dropna=False)
train["Pclass"].value_counts(dropna=False)
train["Age"].value_counts(dropna=False).plot(kind="bar", figsize=(15, 5))
train["Age"].mean()
train["Age"].median()
train[(train["Pclass"] == "3") & (train["Sex"] == "male")]["Age"].mean()
train[train["Pclass"] == "2"]["Age"].median()
train[train["Pclass"] == "3"]["Age"].median()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train = train[~train["Embarked"].isna()]
test = pd.read_csv("/kaggle/input/titanic/test.csv")

age_pclass_1 = train.loc[train["Pclass"] == 1]["Age"].median()
age_pclass_2 = train[train["Pclass"] == 2]["Age"].median()
age_pclass_3 = train[train["Pclass"] == 3]["Age"].median()

train.loc[(train["Pclass"] == 1) & (train["Age"].isna()), "Age"] = age_pclass_1
train.loc[(train["Pclass"] == 2) & (train["Age"].isna()), "Age"] = age_pclass_2
train.loc[(train["Pclass"] == 3) & (train["Age"].isna()), "Age"] = age_pclass_3

test.loc[(test["Pclass"] == 1) & (test["Age"].isna()), "Age"] = age_pclass_1
test.loc[(test["Pclass"] == 2) & (test["Age"].isna()), "Age"] = age_pclass_2
test.loc[(test["Pclass"] == 3) & (test["Age"].isna()), "Age"] = age_pclass_3

fare_pclass_1 = train.loc[train["Pclass"] == 1]["Fare"].median()
fare_pclass_2 = train.loc[train["Pclass"] == 2]["Fare"].median()
fare_pclass_3 = train.loc[train["Pclass"] == 3]["Fare"].median()

train.loc[(train["Pclass"] == 1) & (train["Fare"].isna()), "Fare"] = fare_pclass_1
train.loc[(train["Pclass"] == 2) & (train["Fare"].isna()), "Fare"] = fare_pclass_2
train.loc[(train["Pclass"] == 3) & (train["Fare"].isna()), "Fare"] = fare_pclass_3

test.loc[(test["Pclass"] == 1) & (test["Fare"].isna()), "Fare"] = fare_pclass_1
test.loc[(test["Pclass"] == 2) & (test["Fare"].isna()), "Fare"] = fare_pclass_2
test.loc[(test["Pclass"] == 3) & (test["Fare"].isna()), "Fare"] = fare_pclass_3

age_binner = KBinsDiscretizer(int((train["Age"].max() - train["Age"].min()) / 5), encode="ordinal")
train["Age"] = age_binner.fit_transform(train["Age"].values.reshape(-1, 1))
test["Age"] = age_binner.transform(test["Age"].values.reshape(-1, 1))

fare_binner = KBinsDiscretizer(int((train["Fare"].max() - train["Fare"].min()) / 10), encode="ordinal")
train["Fare"] = fare_binner.fit_transform(train["Fare"].values.reshape(-1, 1))
test["Fare"] = fare_binner.transform(test["Fare"].values.reshape(-1, 1))

bad_features = ["Cabin", "Name", "Ticket", "PassengerId"]
cat_features = ["Pclass", "Sex", "Embarked"]

for cat in cat_features:
    train[cat] = train[cat].astype(str).astype("category")
    test[cat] = test[cat].astype(str).astype("category")

train = train.drop(columns=bad_features, axis=1)
X_train = train.drop(columns=["Survived"], axis=1)
y_train = train["Survived"]
test["Age"].value_counts(dropna=False).plot(kind="bar", figsize=(15, 5))
test["Fare"].value_counts(dropna=False).plot(kind="bar", figsize=(15, 5))
train.dtypes
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
test["Survived"] = clf.predict(test.drop(columns=bad_features, axis=1))
test[["PassengerId", "Survived"]].to_csv("gender_submission.csv")
test["Embarked"].value_counts()
