import os

print(os.listdir("./"))



import pandas as pd # data processing

train = pd.read_csv('../input/train.csv',index_col = "PassengerId")

print(train.shape)

train.head()
test = pd.read_csv('../input/test.csv',index_col = "PassengerId")

print(test.shape)

test.head()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(data = train, x = "Sex", hue = "Survived")
pd.pivot_table(train, index = "Sex", values = "Survived")
sns.countplot(data = train, x = "Pclass", hue = "Survived")
pd.pivot_table(train, index = "Pclass", values = "Survived")
sns.countplot(data = train, x = "Embarked", hue = "Survived")
pd.pivot_table(train, index = "Embarked", values = "Survived")
train.loc[train["Sex"] == "male", "enc_sex"] = 0

train.loc[train["Sex"] == "female", "enc_sex"] = 1



print(train.shape)



train[["Sex","enc_sex"]].head()
test.loc[test["Sex"] == "male", "enc_sex"] = 0

test.loc[test["Sex"] == "female", "enc_sex"] = 1



print(test.shape)



test[["Sex","enc_sex"]].head()
train["Emb_C"] = train["Embarked"] == "C"

train["Emb_S"] = train["Embarked"] == "S"

train["Emb_Q"] = train["Embarked"] == "Q"



print(train.shape)



train[["Embarked","Emb_C","Emb_S","Emb_Q"]].head()
test["Emb_C"] = test["Embarked"] == "C"

test["Emb_S"] = test["Embarked"] == "S"

test["Emb_Q"] = test["Embarked"] == "Q"



print(test.shape)



test[["Embarked","Emb_C","Emb_S","Emb_Q"]].head()
train[train["Fare"].isnull()]
test[test["Fare"].isnull()]
train["fillinFare"] = train["Fare"]



print(train.shape)



train[["Fare","fillinFare"]].head()
test["fillinFare"] = test["Fare"]



print(test.shape)



test[["Fare","fillinFare"]].head()
test.loc[test["Fare"].isnull(), "fillinFare"] = 0

test.loc[test["Fare"].isnull(), ["Fare", "fillinFare"]]
train["Name"].head()
def title(Name):

    Ans = Name.split(', ')[1].split(', ')[0]

    return Ans



train["Name"].apply(title).unique()
train.loc[train["Name"].str.contains("Mr"), "title"] = "Mr"

train.loc[train["Name"].str.contains("Miss"), "title"] = "Miss"

train.loc[train["Name"].str.contains("Mrs"), "title"] = "Mrs"

train.loc[train["Name"].str.contains("Master"), "title"] = "Master"



print(train.shape)



train[["Name", "title"]].head(10)
sns.countplot(data=train, x="title", hue="Survived")
pd.pivot_table(train, index="title", values="Survived")
train["Master"] = train["Name"].str.contains("Master")

print(train.shape)

train[["Name", "Master"]].head(20)
test["Master"] = test["Name"].str.contains("Master")

print(test.shape)

test[["Name", "Master"]].head(20)
train["Child"] = train["Age"] < 14



print(train.shape)



train[["Age", "Child"]].head(10)
test["Child"] = test["Age"] < 14



print(test.shape)



test[["Age", "Child"]].head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1



print(train.shape)



train[["SibSp", "Parch", "FamilySize"]].head()
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1



print(test.shape)



test[["SibSp", "Parch", "FamilySize"]].head()
train["Single"] = train["FamilySize"] == 1



train["Middle"] = (train["FamilySize"] > 1) & (train["FamilySize"] < 5)



train["Big"] = train["FamilySize"] >= 5



print(train.shape)



train[["FamilySize", "Single", "Middle", "Big"]].head(10)
test["Single"] = test["FamilySize"] == 1

test["Middle"] = (test["FamilySize"] > 1) & (test["FamilySize"] < 5)

test["Big"] = test["FamilySize"] >= 5



print(test.shape)



test[["FamilySize", "Single", "Middle", "Big"]].head(10)
feature = ["Pclass", "enc_sex", "Emb_C", "Emb_S", "Emb_Q","fillinFare",

                 "Master","Child", "Single", "Middle", "Big"]

feature
label = "Survived"

label
X_train = train[feature]



print(X_train.shape)



X_train.head()
X_test = test[feature]



print(X_test.shape)



X_test.head()
y_train = train[label]



print(y_train.shape)



y_train.head()
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=9, random_state=0)

model
model.fit(X_train, y_train)
import graphviz

from sklearn.tree import export_graphviz



tree = export_graphviz(model,

                           feature_names=feature,

                           class_names=["Perish", "Survived"],

                           out_file=None)



graphviz.Source(tree)
prediction = model.predict(X_test)



print(prediction.shape)



prediction[0:9]
submission = pd.read_csv('../input/gender_submission.csv',index_col = "PassengerId")



print(submission.shape)



submission.tail(10)
submission["Survived"] = prediction



print(submission.shape)



submission.tail(10)
submission.to_csv("tree.csv")