import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score







for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/titanic/{}.csv".format("train"))

df_test = pd.read_csv("/kaggle/input/titanic/{}.csv".format("test"))

df_gender = pd.read_csv("/kaggle/input/titanic/{}.csv".format("gender_submission"))



#rf = RandomForestClassifier(n_estimators=100)

rf = RandomForestClassifier()
df_train.info()
df_train.describe()
df_train.head(5)
for df in [df_train, df_test]:

    print(df.isna().sum())

    print("\n")
def impute_age_train(row):

    for gender in ["female", "male"]:

        if row["Sex"]==gender:

            for X in range(1,4):

                if row["Pclass"] == X:

                    return df_train.loc[(df_train["Pclass"]==X) & (df_train["Sex"]==gender) & (df_train["Age"].notnull()), "Age"].median()



def impute_age_test(row):

    for gender in ["female", "male"]:

        if row["Sex"]==gender:

            for X in range(1,4):

                if row["Pclass"] == X:

                    return df_test.loc[(df_test["Pclass"]==X) & (df_test["Sex"]==gender) & (df_test["Age"].notnull()), "Age"].median()

                

df_train.loc[(df_train["Age"].isna()), "Age"] = df_train.loc[(df_train["Age"].isna())].apply(impute_age_train, axis=1)

df_test.loc[(df_test["Age"].isna()), "Age"] = df_test.loc[(df_test["Age"].isna())].apply(impute_age_test, axis=1)
print("Number of NaN age in df_train : {}".format(df_train.loc[(df_train["Age"].isna()), "Age"].size))

print("Number of NaN age in df_test : {}".format(df_test.loc[(df_test["Age"].isna()), "Age"].size))
sns.barplot(x="Sex", y="Survived", data=df_train);
sns.barplot(x="Pclass", y="Survived", data=df_train, hue="Sex");
plt.figure(figsize=(16, 10))

sns.violinplot(x="Sex", y="Age", data=df_train, hue="Survived", split="true");
plt.figure(figsize=(8, 5))

sns.countplot(x="Pclass", hue="Sex", data=df_train.loc[(df_train["Survived"]==1)]);
plt.figure(figsize=(8, 5))

sns.countplot(x="Pclass", hue="Sex", data=df_train.loc[(df_train["Survived"]==0)]);
sns.scatterplot(x="Parch", y="Age", hue="Sex", data=df_train);
df_train["Gender"] = df_train["Sex"].astype("category").cat.codes



df_train.sample()
df_test["Gender"] = df_test["Sex"].astype("category").cat.codes



df_test.sample()
selected_columns = ["Gender", "Pclass", "Parch", "SibSp"]



X_train = df_train.loc[:, selected_columns]

X_test = df_test.loc[:, selected_columns]



y_train = df_train.Survived
rf.fit(X_train, y_train)
print(accuracy_score(y_train, rf.predict(X_train)))
rf.predict(X_test)
submission = pd.DataFrame({

    "PassengerId": df_test.PassengerId,

    "Survived": rf.predict(X_test)

})



submission.to_csv("submission.csv", index=False)