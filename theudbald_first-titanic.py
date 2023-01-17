##### Functions

def sex_indexer(sex_element):

    if sex_element == "male":

        return 0

    elif sex_element == "female":

        return 1

    else:

        print("erreur sexe")

        return 2

    

def embarked_indexer(embarked_element):

    if (embarked_element == "Q"):

        return 0

    elif embarked_element == "S":

        return 1

    elif embarked_element == "C":

        return 2

    else:

        return 0



def age_estimator(df):

    age_mean = df[["Age", "Title"]].dropna().groupby("Title").mean()

    return age_mean

    

def title_estimator(name):

    splitted_name = name.split(".")[0]

    title = splitted_name.split(", ")[1]

    return str(title)



def title_mapper(name):

    """

    par importance

    """

    splitted_name = name.split(".")[0]

    title = splitted_name.split(", ")[1]

    if title in ["Mr", "Don", "Dr", "Jonkheer", "Sir"]: # Hommes

        return 0

    elif title in ["Mrs", "Ms", "Dona", "Mme", "the Countess"]: #Femmes

        return 1

    elif title in ["Miss", "Mlle", "Lady"]: # Jeunes

        return 2

    elif title in ["Major", "Col", "Capt"]: # Army

        return 0

    elif title in ["Rev"]: # Sacrifice

        return 0

    elif title in ["Master"]: # Sacrifice

        return 6

    else:

        print("erreur !", title)

    
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



### importing the data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

full_df = pd.concat([train_df.drop("Survived", axis=1), test_df])



### Extract the title

title_serie = full_df["Name"].apply(title_estimator)

title_count = title_serie.value_counts()



train_df["Title"] = train_df["Name"].apply(title_mapper)

test_df["Title"] = test_df["Name"].apply(title_mapper)



### Cleaning NAs

age_mean = age_estimator(train_df)

for i, j in zip(age_mean.index.values, age_mean):

    print(i,j)

train_df["Age"] = train_df["Age"].fillna( train_df["Age"].median())

test_df["Age"] = test_df["Age"].fillna( train_df["Age"].median())

test_df["Fare"] = test_df["Fare"].fillna( train_df["Fare"].median())





### Categorical features

train_df["Sex"] = train_df["Sex"].apply(sex_indexer)

test_df["Sex"] = test_df["Sex"].apply(sex_indexer)



train_df["Embarked"] = train_df["Embarked"].apply(embarked_indexer)

test_df["Embarked"] = test_df["Embarked"].apply(embarked_indexer)
pd.crosstab(train_df["Embarked"], train_df["Survived"])

age_mean = train_df[["Age", "Title"]].dropna().groupby("Title").mean()

print(age_mean)
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.metrics import precision_score

from sklearn.model_selection import cross_val_score



### Preparing the data

X_train = train_df[["Fare", "Sex", "Embarked"]]

y_train = train_df["Survived"]

X_test = test_df[["Fare", "Sex", "Embarked"]]



### Machine Learning

cross_scores = []

train_scores = []

params = []



tr = ExtraTreeClassifier(criterion="gini", max_depth=15)

cross_score = cross_val_score(tr, X_train, y_train, cv=10)

model = tr.fit(X_train, y_train)

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)

train_score = precision_score(y_train, pred_train)

    

print(np.mean(cross_score))

#plt.plot(params, cross_scores, color="r")

#plt.plot(params, train_scores, color="b")

#plt.show()
plt.scatter(train_df["Parch"], train_df["Sex"], c=y_train)

plt.show()
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": pred_test

    })

submission.to_csv('titanic_3.csv', index=False)
train_df.head()