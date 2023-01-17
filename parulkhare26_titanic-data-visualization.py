import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

IDtest = test["PassengerId"]
train_len = len(train)

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

dataset.isnull().sum()

train.info()

train.isnull().sum()
train.head()
# Summarie and statistics

train.describe()
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")


g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")


g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")


g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
dataset["Fare"].isnull().sum()


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")


dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
train[["Sex","Survived"]].groupby('Sex').mean()


g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset["Embarked"].isnull().sum()


dataset["Embarked"] = dataset["Embarked"].fillna("S")
 

g = sns.factorplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
 

g = sns.factorplot("Pclass", col="Embarked",  data=train,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")


dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)