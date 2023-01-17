import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

plt.rcParams["figure.figsize"] = (10, 8)

sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
train.shape, test.shape
train.columns.values, test.columns.values
set(train.columns.values) - set(test.columns.values)
train.info()
train.dtypes
train.isnull().sum()
train_survived = train["Survived"].value_counts()

train_survived
train_s_rate = train_survived[1] / train_survived.sum()

train_s_rate
train_survived.plot(kind="bar")

plt.title("Survival Histogram - {:0.1%} Survived (All Passengers)".format(train_s_rate))

plt.ylabel("Number of Passengers")

plt.xlabel("0 = Perished, 1 = Survived")

plt.show()
train["Survived"].value_counts(normalize=True).plot(kind="bar")



plt.title("Survival Histogram")

plt.ylabel("% of Passengers")

plt.xlabel("0 = Perished, 1 = Survived")



plt.show()
fig, axs = plt.subplots(1,3, figsize=(20,4))



for i, f in enumerate(["Fare", "Age", "Pclass"]):

    sns.distplot(train[f].dropna(), kde=False, ax=axs[i]).set_title(f)

    axs[i].set(ylabel="# of Passengers")



plt.suptitle("Feature Histograms (Ignoring Missing Values)")

plt.show()
fig, axs = plt.subplots(1,2, figsize=(20,4))



for i, f in enumerate(["Parch", "SibSp"]):

    sns.distplot(train[f].dropna(), kde=False, ax=axs[i]).set_title(f)

    axs[i].set(ylabel="# of Passengers")



plt.suptitle("Feature Histograms (Ignoring Missing Values)")

plt.show()
sns.heatmap(train.corr(), annot=True, cmap="coolwarm")

plt.show()
train.corr()["Survived"].sort_values()
fig, axs = plt.subplots(1, 2, figsize=(12,6))

for i, sex in enumerate(["female", "male"]):

    p = train[train["Sex"] == sex]["Survived"].value_counts(normalize=True).sort_index().to_frame().reset_index()

    sns.barplot(x=["Perished", "Survived"], y="Survived", data=p, hue="index", ax=axs[i], dodge=False)

    axs[i].set_title("Survival Histogram - {:0.1%} Survived ({})".format(p.loc[1,"Survived"], sex))

    axs[i].set_ylabel("Survival Rate")

    axs[i].get_legend().remove()
sex_map = {"male": 0, "female": 1}

sex_encoded = train["Sex"].map(sex_map)



sex_df = pd.DataFrame({"Sex_Encoded": sex_encoded, "Survived": train["Survived"]})

sex_df.corr()
n_male = len(train[train["Sex"] == "male"])

n_female = len(train[train["Sex"] == "female"])

"Males: {:.1%}, Females: {:.1%}".format(n_male / len(train), n_female / len(train))
fig, axs = plt.subplots(1, 3, figsize=(20,4))



for i, pclass in enumerate([1, 2, 3]):

    p = train[train["Pclass"] == pclass]["Survived"].value_counts(normalize=True)

    p.plot(kind="bar", ax=axs[i])

    axs[i].set_title("Survival Histogram - {:0.1%} Survived (Class {})".format(p[1], pclass))

    axs[i].set_ylabel("% of Passengers")

    axs[i].set_xlabel("0 = Perished, 1 = Survived")
g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar")

g.despine(left=True)

g.set_ylabels("Survival Probability")

plt.show()
g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar")

g.despine(left=True)

g.set_ylabels("Survival Probability")

plt.show()
g = sns.catplot(x="Embarked", y="Survived", data=train, kind="bar")

g.despine(left=True)

g.set_ylabels("Survival Probability")

plt.show()
g = sns.catplot("Pclass", col="Embarked",  data=train, kind="count", palette="muted")

g = g.set_ylabels("# of Passengers")
g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar")

g.despine(left=True)

g.set_ylabels("Survival Probability")

plt.show()
g = sns.kdeplot(train["Age"][train["Survived"] == 0], label="Perished", shade=True, color=sns.xkcd_rgb["pale red"])

g = sns.kdeplot(train["Age"][train["Survived"] == 1], label="Survived", shade=True, color=sns.xkcd_rgb["steel blue"])

plt.xlabel("Age")

plt.ylabel("Density")

plt.show()
g = sns.FacetGrid(train, col="Survived")

g = g.map(sns.distplot, "Age")
child_ages = [13, 14, 15, 16, 17, 18, 19, 20, 21]

num_children = []

for i in child_ages:

    num_children.append(len(train[train["Age"] < i]))

                        

g = sns.barplot(x=child_ages, y=num_children)

plt.ylabel("# of Passengers")

plt.xlabel("Age Threshold")

plt.title("Number of Passengers Under Certain Age Thresholds")

plt.show()