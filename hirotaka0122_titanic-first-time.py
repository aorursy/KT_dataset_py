import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df  = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



pd.options.display.precision = 3    # 桁数の設定
train_df.head()
train_df.tail()
test_df.head()
test_df.tail()
print("rate of surveid : {:.4f}".format(train_df["Survived"].sum() / len(train_df)))
print("Train")

print(train_df.info())



print("-"*50)



print("Test")

print(test_df.info())
train_df["gender"] = train_df["Sex"].map(lambda x: 0 if x == "male" else 1)

test_df["gender"] = test_df["Sex"].map(lambda x: 0 if x == "male" else 1)
train_df["port"] = train_df["Embarked"].map(lambda x: 0 if x == "C" else 1 if x == "Q" else 2 if x == "S" else np.nan)

test_df["port"] = test_df["Embarked"].map(lambda x: 0 if x == "C" else 1 if x == "Q" else 2 if x == "S" else np.nan)



train_df.head()

train_df.describe()
print(train_df.isnull().sum())

print("-"*50)

print(test_df.isnull().sum())
train_df = train_df.drop(["Cabin", "Sex", "Embarked"], axis=1)

test_df = test_df.drop(["Cabin", "Sex", "Embarked"], axis=1)
train_df[["Pclass", "Survived"]].groupby(["Pclass"]).mean().sort_values(by="Survived", ascending=False)
sns.countplot(x="Pclass", data=train_df)
train_df[["gender", "Survived"]].groupby(["gender"]).mean().sort_values(by="Survived", ascending=False)
sns.countplot(x="gender", data=train_df)
gen = sns.FacetGrid(train_df, col="Survived")

gen.map(plt.hist, "gender", bins=2)
train_df[["SibSp", "Survived"]].groupby(["SibSp"]).mean().sort_values(by="Survived", ascending=False)
sns.countplot(x="SibSp", data=train_df)
train_df[["Parch", "Survived"]].groupby(["Parch"]).mean().sort_values(by="Survived", ascending=False)
sns.countplot(x="Parch", data=train_df)
train_df["family"] = train_df["SibSp"] + train_df["Parch"]

test_df["family"] = test_df["SibSp"] + test_df["Parch"]

train_df.head()
train_df[["family", "Survived"]].groupby(["family"]).mean().sort_values(by="Survived", ascending=False)
sns.countplot(x="family", data=train_df)
age = sns.FacetGrid(train_df, col="Survived")

age.map(plt.hist, "Age", bins=10)
for dataset in [train_df, test_df]:

    dataset["honorifics"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
pd.crosstab(train_df["honorifics"], train_df["gender"])
for dataset in [train_df, test_df]:

    dataset["honorifics"] = dataset["honorifics"].replace(["Lady", "Countess","Capt", "Col",

                                                 "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")



    dataset["honorifics"] = dataset["honorifics"].replace("Mlle", "Miss")

    dataset["honorifics"] = dataset["honorifics"].replace("Ms", "Miss")

    dataset["honorifics"] = dataset["honorifics"].replace("Mme", "Mrs")
train_df[["honorifics", "Survived"]].groupby(["honorifics"], as_index=False).mean()
for_map_dict = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}



for dataset in [train_df, test_df]:

    dataset["honorifics"] = dataset["honorifics"].map(for_map_dict)
train_df.head()
train_df = train_df.drop(["Name", "PassengerId", "SibSp", "Parch", "Ticket"], axis=1)

test_df = test_df.drop(["Name", "PassengerId", "SibSp", "Parch", "Ticket"], axis=1)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
print(train_df[train_df["port"].isnull()])

print("-"*50)

train_df[train_df["Fare"] <= 80.0].groupby("port").count()
train_df["port"] = train_df["port"].fillna(2.0)
test_df[test_df["Fare"].isnull()]
test_df["Fare"] = test_df["Fare"].fillna(0)
print(train_df.isnull().sum())



print("-"*50)



print(test_df.isnull().sum())
from sklearn.model_selection import train_test_split



X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]



x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



model = DecisionTreeClassifier()

model.fit(x_train, y_train)



print(accuracy_score(y_val, model.predict(x_val)))
submit = pd.read_csv("../input/test.csv")

submit = submit.drop(submit.columns.values[1:], axis=1)



result = pd.DataFrame(model.predict(test_df.values))

submit["Survived"] = result



submit.to_csv("submission.csv", index=False)