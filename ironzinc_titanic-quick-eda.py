import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

%matplotlib inline

sns.set()
data_df = pd.read_csv("../input/titanic/train.csv")

data_df.head()
data_df.describe(include="all")
sns.countplot(x="Survived", data=data_df)
sns.countplot(x="Pclass", data=data_df)
sns.countplot(x="Sex", data=data_df)
sns.distplot(a=data_df["Age"].dropna())
sns.countplot(x="SibSp", data=data_df)
sns.countplot(x="Parch", data=data_df)
sns.countplot(x="Embarked", data=data_df)
sns.distplot(a=data_df["Fare"])
sns.boxplot(x="Fare", data=data_df)
sns.heatmap(data_df.corr(), annot=True)
sns.violinplot(x="Pclass", y="Age", hue="Sex", data=data_df)
sns.swarmplot(x="Pclass", y="Fare", data=data_df)
sns.countplot(x="Pclass", hue="Survived", data=data_df)
sns.countplot(x="Sex", hue="Survived", data=data_df)
grouped_df = data_df.groupby("Survived")

survived_df = grouped_df.get_group(1)



age_survived = survived_df["Age"].dropna()

sns.distplot(a=age_survived, label="Survived", bins=range(0, 80, 5), kde=False)

sns.distplot(a=data_df["Age"].dropna(), label="Total", bins=range(0, 80, 5), kde=False)

plt.legend()
sibsp_survived = survived_df["SibSp"]

sns.distplot(a=sibsp_survived, label="Survived", bins=range(0, 8, 1), kde=False)

sns.distplot(a=data_df["SibSp"], label="Total", bins=range(0, 8, 1), kde=False)

plt.legend()
parch_survived = survived_df["Parch"]

sns.distplot(a=parch_survived, label="Survived", bins=range(0, 6, 1), kde=False)

sns.distplot(a=data_df["Parch"], label="Total", bins=range(0, 6, 1), kde=False)

plt.legend()
family_survived = survived_df["Parch"] + survived_df["SibSp"]

sns.distplot(a=family_survived, label="Survived", bins=range(0, 11, 1), kde=False)

sns.distplot(a=data_df["Parch"]+data_df["SibSp"], label="Total", bins=range(0, 11, 1), kde=False)

plt.legend()
fare_survived = survived_df["Fare"]

sns.distplot(a=fare_survived, label="Survived", bins=range(0, 200, 10), kde=False)

sns.distplot(a=data_df["Fare"], label="Total", bins=range(0, 200, 10), kde=False)

plt.legend()
sns.countplot(x="Embarked", hue="Survived", data=data_df)
sns.countplot(x="Embarked", hue="Pclass", data=data_df)
data_df.isnull().sum()
data_df["Age"] = data_df["Age"].fillna(np.mean(data_df["Age"]))

data_df["Embarked"] = data_df["Embarked"].fillna("S")

data_df["Cabin"] = data_df["Cabin"].fillna("Other")

data_df.isnull().sum().sum()
data_df
data_df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

data_df["Sex"] = LabelEncoder().fit_transform(data_df["Sex"])

data_df["Cabin"] = LabelEncoder().fit_transform(data_df["Cabin"])

data_df["Embarked"] = LabelEncoder().fit_transform(data_df["Embarked"])
rf = RandomForestClassifier(n_estimators=500)

et = ExtraTreesClassifier(n_estimators=500)



rf.fit(data_df.drop("Survived", axis=1), data_df["Survived"])

et.fit(data_df.drop("Survived", axis=1), data_df["Survived"])
figures, axes = plt.subplots(1, 2, figsize=(24, 12))

print(rf.feature_importances_)

axes[0].bar(height=rf.feature_importances_, x=range(1, 9), tick_label=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])

axes[0].set_title("Random Forest Feature Importances")

axes[1].bar(height=et.feature_importances_, x=range(1, 9), tick_label=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])

axes[1].set_title("Extra Trees Feature Importances")