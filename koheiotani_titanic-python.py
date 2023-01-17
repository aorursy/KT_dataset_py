# data analysis and wrangle

import pandas as pd 

import numpy as np

from collections import Counter



# visualisation 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# step1: load data 

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

combine = [train_data,test_data]
# divide data into categorical and numeric variables

# -> so we know how to visualize them later

train_data.head()

#print (pd.unique(train_data.loc[:,"Embarked"]))

#print (pd.unique(test_data.loc[:,"Embarked"]))

# categorical variable

#     nominal var: PassengeId[int], Name[char], Sex[male/female], Ticket[char+int], cabin[NaN]

#     ordinal var: Pclass[int], Embarked[char]

# numerical variable

#     discrete var: Age, SibSp, Parch

#     continuous var: Fare

# Label

#     survived: 0 -- not survived, 1 -- survived
# any null, outlier or abnormal feature or pattern or representative

# -> the reason why we need to find them is because

# -> ML models do not like missing values and will be confused with outlier

#    and also if features are representative, it might contribute a lot for our model

print (train_data.info())

print ("-"*40)

print (test_data.info())

# train - cabin(a lot of null values -> think if drop them.)

#       - Embarked (2 null values -> replace with similar values)

#       - Age (200+ null values --> replace with similar values)

# test - cabin (a lot of null values -> think if drop them.)

#      - Fare (1 null values -> replace with median or mean)
# check representativeness of features

# my purpose: if the feature is representative (-> feature is useful)

#             e.g. like Pclass, if Pclass = 1 has high survival (> average), and others have lower rates, 

#             then this feature may have correlations with survival rates --> so keep it. 

train_data.head()

train_data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

## PassengerId --> unique ID from 1 to 891 --> not representative --> drop them

# Survived --> 38% survival rate 

# Pclass --> most of people (50%+) from Pclass 3 --> survial rate for different Pclass

# Age --> has missing values & 80% people below 41 years old --> survival rate for different ages

# SibSp --> most of people (60%+) does not have brothers & sisters together -> look correlations

# Parch --> more than 70% does have parents together -> look correlations

# Fare --> most of people (80%) pay below 39 but there are weired values like 512(?) & 0(staff).

#      --> see correlations first & may build new ratio variables based on it

#      --> too many unique Fares --> need to build ratio variables
train_data.describe(include =["O"])

# Name -> unique -> it is useless now --> may group similar name together

# Sex -> most of people (577/891) are male --> see correlations

# Ticket -> some ticket number are same --> why?

## Cabin -> some Cabin are same -> some people share one room

#        -> not only has too many missing value and also it might have no correlations with labels

# Embarked -> people from three ports & 644/889 from S
# Assumption

# group one: high level customers --> [Pclass, Fare] --> high survival rate

# group two: have relatives on boat --> [SibSp, Parch] --> high survival rate

# group three: lady first --> Sex --> lady has high survival rate

# group four: child, elder first --> Age --> have high survival rate

# group five: from different ports --> Embarked --> surival rate varying
# group one - true

train_data[["Pclass", "Survived"]].groupby("Pclass", as_index = False).mean().sort_values(by = "Survived", ascending = False)

#train_data[["Fare", "Survived"]].groupby("Fare", as_index = False).mean().sort_values(by = "Survived", ascending = False)

# try later about Fare - maybe make a range variable
#g = sns.FacetGrid(train_data)

#g.map(sns.pointplot, "Pclass", "Survived", palette='deep')
# group two - true

print (train_data[["SibSp", "Survived"]].groupby("SibSp", as_index = False).mean().sort_values(by = "Survived", ascending = False))

print ("-"*40)

print (train_data[["Parch", "Survived"]].groupby("Parch", as_index = False).mean().sort_values(by = "Survived", ascending = False))

# it seems like when you have too high or no SibSp (>=3 or =0) -> below average survival rate

# it seems like when you have too high or no Parch (>=4 or =0) -> below average survival rate

# if you have too many family memeber, you dont have enough power to help every one and they too

# and also, if you do have family memeber, nobody help you

# also, maybe we can combine those two variables later
# group three - true

train_data[["Sex", "Survived"]].groupby("Sex", as_index = False).mean().sort_values(by = "Survived", ascending = False)
# group four - later -> need to re-design variable

g = sns.FacetGrid(train_data, col = "Survived")

g.map(plt.hist, "Age")

# not obvious -> create Age range

train_data["AgeRange"] = pd.cut(train_data["Age"], 5)

train_data[["AgeRange", "Survived"]].groupby("AgeRange", as_index = False).mean().sort_values(by = "Survived", ascending = False)
# group five - true assumption

train_data[["Embarked", "Survived"]].groupby("Embarked", as_index = False).mean().sort_values(by = "Survived", ascending = False)

# people from C port have high survial rate
train_data = pd.read_csv("../input/train.csv")

train_label = train_data["Survived"]

train_data = train_data.drop("Survived", axis = 1)

test_data = pd.read_csv("../input/test.csv")

combine = [train_data,test_data]

train_data.head()
# complete incomplete features

# train 

#       - Embarked (2 null values -> replace with similar values)

#       - Age (200+ null values --> replace with similar values)

# test 

#      - Fare (1 null values -> replace with median or mean)



# correct - if outlier drop them / remove abnormal data

#         - PassengerId[int] ————— drop 

#         - Name[char] ————— drop

#         - cabin[NaN] ————— drop



# convert categorical variables into numeric variables/dummy variables/one-hot

#         - Sex[male/female] ————— dummy 

#         - Ticket[char+int] ————— dummy 

#         - Pclass[int] ————— dummy 

#         - Embarked[char] ————— dummy 



# create  - build new variables based on existing varialbes
# correct features

print ("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_df = train_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1)

test_df = test_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1)

del train_data 

del test_data

combine = [train_df, test_df]



print ("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# convert features

categories_to_dummies = ["Sex", "Pclass", "Embarked"]

for i, df in enumerate(combine):

    for j in categories_to_dummies:

        # train

        # categories to dummies 

        tmp = pd.get_dummies(df[j], prefix=j)

        df = df.join(tmp)

        df = df.drop(j, axis = 1)

        combine[i] = df

del tmp

del df

print ("After", combine[0].shape, combine[1].shape)

combine[0].head()
# handle missing value

# Fare - test 

# better solution

# test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

tmp_test = combine[1]

tmp_test.loc[tmp_test["Fare"].isnull(), "Fare"] = tmp_test["Fare"].median()

pd.isnull(tmp_test["Fare"]).sum() > 0
train_df["AgeRange"] = pd.cut(train_df["Age"], 5)
for i, df in enumerate(combine):

    df.loc[df["Age"] <= 16.336, "Age"] = 0

    df.loc[(df["Age"] > 16.336) & (df["Age"] <=32.252), "Age"] = 1

    df.loc[(df["Age"] > 32.252) & (df["Age"] <=48.168), "Age"] = 2

    df.loc[(df["Age"] > 48.168) & (df["Age"] <=64.084), "Age"] = 3

    df.loc[(df["Age"] > 64.084) & (df["Age"] <=80.0), "Age"] = 4

    combine[i] = df

del df

print (combine[0]["Age"])

print (combine[1]["Age"])
# clustering

# build temporary train & test sets

tmp_train = combine[0].copy()

tmp_test = combine[1].copy()

tmp_train = tmp_train.drop("Age", axis = 1)

tmp_test = tmp_test.drop("Age", axis = 1)

# kmeans clustering for train & test sets

# & assign "kmeans_labels" back to "combine sets"

kmeans = KMeans(n_clusters=5, random_state=0).fit(tmp_train)

combine[0]["kmeans_labels"] = pd.Series(kmeans.labels_)

combine[1]["kmeans_labels"] = pd.Series(kmeans.predict(tmp_test))

# verify success or not

combine[1].head()
# replace age's missing value 

tmp_train = combine[0].copy()

tmp_test = combine[1].copy()

tmp_age_train = []

tmp_age_test = []

for i in range(5):

    tmp_age_train.append(Counter(tmp_train.loc[tmp_train["kmeans_labels"] == i, "Age"]).most_common()[0][0])

    tmp_age_test.append(Counter(tmp_test.loc[tmp_test["kmeans_labels"] == i, "Age"]).most_common()[0][0])

for j in range(5):

    tmp_train.loc[(tmp_train["kmeans_labels"] == j) & (tmp_train["Age"].isnull()), "Age"] = tmp_age_train[j]

    tmp_test.loc[(tmp_test["kmeans_labels"] == j) & (tmp_test["Age"].isnull()), "Age"] = tmp_age_test[j]

combine[0]["Age"] = tmp_train["Age"]

combine[1]["Age"] = tmp_test["Age"]

print (pd.isnull(combine[0]["Age"]).sum() >0, pd.isnull(combine[1]["Age"]).sum() >0)
# modeling

# drop, kmeans, since it is not original features

X_train = combine[0].drop("kmeans_labels", axis = 1)

Y_train = train_label

X_test = combine[1].drop("kmeans_labels", axis = 1)

print ("After", X_train.shape, Y_train.shape, X_test.shape)

X_train.head()
# logistic regression

logistic = LogisticRegression()

logistic.fit(X_train, Y_train)

#logistic.predict(X_test)

print ("train_acc", logistic.score(X_train, Y_train)*100)