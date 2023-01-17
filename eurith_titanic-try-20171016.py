# coding: utf-8

#What I'm trying to do

#1: Before generating the feature quantity, confirm the specific data and decide the policy.

#2: Generate feature quantities from each item and fill in missing values other than age. Ticket is excluded this time. Unnecessary feature quantities are appropriately reduced.

#3: The algorithm uses a random forest. Use the grid search to get the best parameters.

#4: Make data for submission using the best result parameter.



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

test_result = pd.read_csv("../input/genderclassmodel.csv")



#For data processing, make train, test and answer all together and make over 1300 data sets. (To divide later)



test = pd.merge(test_data, test_result, how="outer", on="PassengerId")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

df.head()
#1: Before generating the feature quantity, confirm the specific data and decide the policy.



#Confirm about Parch

g = sns.factorplot(y="Survived", x="Parch", data=df, kind="bar")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")



#High survival rate 1, 2, 3. 9 may be abnormal value, so confirmation is necessary.

df[df["Parch"]==9].head()
#Confirm about SibSp

g = sns.factorplot(y="Survived", x="SibSp", data=df, kind="bar")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")



#1 and 2 have a high survival rate. The survival rate decreases as the number increases.
#2: Generate feature quantities from each item and fill in missing values other than age. Ticket is excluded this time. Unnecessary feature quantities are appropriately reduced.



df = pd.get_dummies(df, columns=["Embarked"], prefix="Em")

df = pd.get_dummies(df, columns=["Pclass"], prefix="Pc")

df = pd.get_dummies(df, columns=["Sex"])

df.drop(labels=["Ticket","Sex_male"], axis=1, inplace=True)

df.head()
#Detects the title from the name and converts it into the feature quantity

df_title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]

df["Title"] = pd.Series(df_title)

#df["Title"].value_counts() #Because Mlle looked like Mile and made a spelling mistake, for confirmation

g = sns.countplot(x="Title", data=df)

g = plt.setp(g.get_xticklabels(), rotation=45)
df["Title"] = df["Title"].replace(['the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms":1, "Mme":1, "Mlle":1, "Lady":1, "Mrs":2, "Mr":3, "Rare":4})

df["Title"] = df["Title"].astype('int')

df.drop(labels=["Name"], axis=1, inplace=True)

df.head()
#What is the distribution when adding Parch and SibSp? (+1 at the end adds the boat person himself / herself)

df["Fsize"] = df["Parch"] + df["SibSp"] + 1

g = sns.factorplot(y="Survived", x="Fsize", data=df, kind="bar")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")



#High numbers outstanding 2, 3 and 4. I want to divide it into 4 groups of "single person", "small family", "middle family", "large family"

df["F_single"] = df["Fsize"].map(lambda s: 1 if s == 1 else 0)

df["F_small"] = df["Fsize"].map(lambda s: 1 if 2 <= s <= 4 else 0)

df["F_middle"] = df["Fsize"].map(lambda s: 1 if 5 <= s <= 7 else 0)

df["F_large"] = df["Fsize"].map(lambda s: 1 if 8 <= s else 0)



df.drop(labels=["Parch","SibSp","Fsize"], axis=1, inplace=True)



df.head()
#Confirm the data of the person corresponding to the missing value of Fare. Estimate death from the condition "3rd cabin, male, 60.5 years old" and fill it with 0

df["Fare"] = df["Fare"].fillna(0)

df.head()
#The cabin is taken from the initials of Cabin and in the case of Nan it is X

df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df["Cabin"]])

g = sns.factorplot(y="Survived", x="Cabin", data=df, kind="bar", order=["A","B","C","D","E","F","G","T","X"])



#Code for creating feature quantity with low survival rate X as 0 and others as 1 (not used because score decreased by about 0.015 Reference code)

#df["Cabin"] = df["Cabin"].map({"A":1, "B":1, "C":1, "D":1, "E":1, "F":1, "G":1, "T":1, "X":0})

#df["Cabin"] = df["Cabin"].astype("int")



df = pd.get_dummies(df, columns=["Cabin"], prefix="Cab")

#df.drop(labels="Cab_1", axis=1, inplace=True)

df.head()
#Missing values of age are filled with linear regression



df_age_train = pd.DataFrame(df[:])

df_age_train = df_age_train.dropna()

df_age_train_X = pd.DataFrame(df_age_train[:])



df_age_train_Y = pd.Series(df_age_train_X["Age"])

df_age_train_X.drop(labels=["Age"], axis=1, inplace=True)



df_age_test_X = pd.DataFrame(df[df["Age"].isnull()])

df_age_test_X.drop(labels=["Age"], axis=1, inplace=True)
from sklearn.linear_model import LinearRegression



A_clf = LinearRegression()

A_clf.fit(df_age_train_X, df_age_train_Y)



ID_test_age = df_age_test_X["PassengerId"].reset_index(drop=True)

age_pred = pd.Series(A_clf.predict(df_age_test_X), name="Age")

df_age_test = pd.concat([ID_test_age, age_pred], axis=1)



df_age_test = pd.merge(df_age_test, df_age_test_X, how="outer", on="PassengerId")



df = pd.concat([df_age_train, df_age_test], axis=0)

df = df.sort_values(['PassengerId'], ascending=[True]).reset_index(drop=True)

df["Age"] = df["Age"].where(df["Age"] < 0, 1)

df["Age"] = df["Age"].astype('int')
#Repartition the data set back to the training set and test set.



X_train = pd.DataFrame(df[:len(train)])

X_test = pd.DataFrame(df[len(train):])



Y_train = pd.Series(X_train["Survived"])

X_train.drop(labels=["Survived","PassengerId"], axis=1, inplace=True)



X_test.drop(labels=["Survived","PassengerId"], axis=1, inplace=True)
#3: The algorithm uses a random forest. Use the grid search to get the best parameters.



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
RFC = RandomForestClassifier()

RF_Param_Grid = {

    "max_depth": [4,8,16,32],

    "min_samples_split": [2,4,8,16],

    "min_samples_leaf": [1,3],

    "bootstrap": [False],

    "n_estimators": [50,100],

    "criterion": ["gini"]

}

kfold = StratifiedKFold(n_splits=10)



g_clf = GridSearchCV(RFC, param_grid=RF_Param_Grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

g_clf.fit(X_train, Y_train)
g_clf.best_score_
g_clf.best_params_
#4: Make data for submission using the best result parameter.



bp = g_clf.best_params_



clf = RandomForestClassifier(

    bootstrap=bp["bootstrap"], 

    criterion=bp["criterion"], 

    max_depth=bp["max_depth"], 

    min_samples_leaf=bp["min_samples_leaf"], 

    min_samples_split=bp["min_samples_split"], 

    n_estimators=bp["n_estimators"]

)

clf.fit(X_train, Y_train)
IDtest = test_data["PassengerId"]

test_Survived = pd.Series(clf.predict(X_test), name="Survived")

results = pd.concat([IDtest, test_Survived], axis=1)