# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas import Series, DataFrame
# find correlation between sex and survival rate

sns.catplot(x='Sex', y = 'Survived', hue='Sex', data=train_data, kind='bar')
sns.countplot(x="Sex", hue="Sex", data=train_data)
sns.countplot(x="Pclass", data=train_data, hue="Sex")
def male_female_child(passenger):

    age, sex = passenger

    if age < 16:

        return "child"

    else:

        return sex
train_data["person"] = train_data[["Age", "Sex"]].apply(male_female_child, axis=1)
train_data.head(10)
sns.countplot(x="Pclass", data=train_data, hue="person")
train_data["Age"].hist(bins=70)
train_data["Age"].mean()
train_data["person"].value_counts()
# Age distribution against gender 

fig = sns.FacetGrid(train_data, hue="Sex", aspect=4)

fig.map(sns.kdeplot, "Age", shade=True)

oldest = train_data["Age"].max()

fig.set(xlim=(0, oldest))

fig.add_legend()
# Age distribution vs different passenger classes

fig = sns.FacetGrid(train_data, hue="Pclass", aspect=4)

fig.map(sns.kdeplot, "Age", shade=True)

oldest = train_data["Age"].max()

fig.set(xlim=(0, oldest))

fig.add_legend()
deck = train_data["Cabin"].dropna()

deck.head()
levels = []



for level in deck:

    levels.append(level[0])

cabin_df = DataFrame(levels)

cabin_df.columns = ["Cabin"]

cabin_df.head()
sns.countplot(x="Cabin", data=cabin_df, palette="winter_d")
cabin_df = cabin_df[cabin_df.Cabin != "T"]

sns.countplot(x="Cabin", data=cabin_df, palette="summer")
train_data.head()
sns.countplot(x="Embarked", data=train_data, hue="Pclass", palette="winter")
# who was alone and who was with family?

train_data["Alone"] = train_data.SibSp + train_data.Parch

train_data["Alone"].head()
train_data["Alone"].loc[train_data["Alone"] > 0] = "With Family"

train_data["Alone"].loc[train_data["Alone"] == 0] = "Alone"
train_data.head()
sns.countplot(x="Alone", data=train_data, hue="person")
train_data["Survivor"] = train_data.Survived.map({0:"No", 1:"Yes"})

sns.countplot(x="Survivor", data=train_data, palette="Set1")
sns.catplot(x="Pclass", y="Survived", hue="person", data=train_data, kind="point")
sns.lmplot(x="Age", y="Survived", data=train_data)
sns.lmplot(x="Age", y="Survived", hue="Pclass", data=train_data, palette="Set1")
generations = [10, 20, 30, 40, 50, 60, 80]

sns.lmplot(x="Age", y="Survived", hue="Pclass", data=train_data, x_bins=generations, palette="Set1")
sns.lmplot(x="Age", y="Survived", hue="Sex", data=train_data, x_bins=generations, palette="Set1")
sns.lmplot(x="Age", y="Survived", hue="Alone", data=train_data, x_bins=generations, palette="Set1")
train_data["person"].loc[train_data["person"] == "female"] = 0

train_data["person"].loc[train_data["person"] == "male"] = 1

train_data["person"].loc[train_data["person"] == "child"] = 2
train_data.head()
train_data.person.fillna(0, inplace=True)

train_data.Age.fillna(29.0, inplace=True)
# Apply Random Forest Classifier to Predict

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)



y = train_data["Survived"]

features = ["Pclass", "person", "Age", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

clf.fit(X, y)
test_data["person"] = train_data[["Age", "Sex"]].apply(male_female_child, axis=1)
test_data["person"].loc[test_data["person"] == "female"] = 0

test_data["person"].loc[test_data["person"] == "male"] = 1

test_data["person"].loc[test_data["person"] == "child"] = 2
test_data.head()
test_data.info()
test_data.Age.fillna(29.0, inplace=True)
test_data.info()
X_test = test_data[features]

y_predict = clf.predict(X_test)
Output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": y_predict})

Output.to_csv("my_submission_titanic.csv", index=False)