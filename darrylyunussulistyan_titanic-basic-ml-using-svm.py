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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

df_train = pd.read_csv("../input/titanic/train.csv")
df_train.shape
df_train.describe(include='all')
df_test.shape
df_test.describe(include='all')
import matplotlib.pyplot as plt

import seaborn as sns

import math
sns.swarmplot(data=df_train, x="Sex", y="Age", hue="Survived")

plt.legend()

plt.show()
sns.countplot(x="Pclass", hue="Survived", data=df_train)

plt.legend()

plt.show()
sns.distplot(df_train[df_train["Survived"] == 1].Fare.dropna(), bins=20, kde=False, label="survived")

sns.distplot(df_train[df_train["Survived"] == 0].Fare.dropna(), bins=20, kde=False, label="dead")

plt.title("Fare")

plt.legend()



plt.show()
plt.figure(figsize=(20,5))



plt.subplot(121)

sns.countplot(x="SibSp", hue="Survived",  data=df_train)

plt.title("SibSp")

plt.legend()



plt.subplot(122)

sns.countplot(x="Parch", hue="Survived",  data=df_train)

plt.title("Parch")

plt.legend()



plt.show()
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]

sns.countplot(x="FamilySize", hue="Survived",  data=df_train)

plt.title("FamilySize vs Survived")

plt.legend()

plt.show()
sns.countplot(data=df_train, x="Embarked", hue="Pclass")

plt.title("Embarked")

plt.legend()

plt.show()
plt.figure(figsize=(20,5))



plt.subplot(131)

sns.countplot(data=df_train, x="Embarked", hue="Survived")

plt.title("Embarked")

plt.legend()



plt.subplot(132)

sns.countplot(data=df_train[df_train["Sex"] == 'male'], x="Embarked", hue="Survived")

plt.title("Embarked male")

plt.legend()



plt.subplot(133)

sns.countplot(data=df_train[df_train["Sex"] == 'female'], x="Embarked", hue="Survived")

plt.title("Embarked female")

plt.legend()



plt.show()
sns.catplot(x="Pclass", y="Survived",

            hue="Sex", row="Embarked",

            data=df_train,

            kind="point")

plt.show()
def getTitleFromName(strName):

    givenName = strName.split(",")[1]

    title = givenName.split(".")[0].lower().strip()

    return title
plt.figure(figsize=(20, 5))

df_train["Title"] = df_train["Name"].apply(getTitleFromName)

sns.countplot(x="Title", hue="Survived", data=df_train)

plt.legend()

plt.show()
df_train["Title"] = df_train["Name"].apply(getTitleFromName)

df_train["Title"].value_counts()
df_test["Title"] = df_test["Name"].apply(getTitleFromName)

df_test["Title"].value_counts()
def getTitleFromNameBigOnly(strName):

    titleMapBigOnly = {

        "mr": 1,

        "miss": 2,

        "mrs": 3,

        "master": 4

    }

    title = getTitleFromName(strName)

    if title in titleMapBigOnly:

        return title

    else:

        return 'other'
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
def stdScale(X_train, X_test, columnName):

    fsStdScaler = StandardScaler()

    fsStdScaler = fsStdScaler.fit(X_train[[columnName]])

    X_train.loc[:,[columnName]] = fsStdScaler.transform(X_train[[columnName]])

    X_test.loc[:,[columnName]] = fsStdScaler.transform(X_test[[columnName]])

    return X_train, X_test
def labelEncode(X_train, X_test, columnName):

    enc = LabelEncoder()

    enc = enc.fit(X_train[columnName])

    X_train.loc[:,[columnName]] = enc.transform(X_train[columnName])

    X_test.loc[:,[columnName]] = enc.transform(X_test[columnName])

    return X_train, X_test
def medianPerTitle(X_train):

    hm = {}

    for entry in X_train[["Title", "Age"]].dropna().values:

        title = entry[0]

        age = entry[1]

        if title in hm:

            hm[title].append(age)

        else:

            hm[title] = [age]

    

    for title in hm:

        hm[title] = np.median(hm[title])

    

    print(hm)



    return hm
def rfattempt1_transform(X_train, X_test, y_train, y_test=None):

    # encode Sex. Sex is not encoded using OHE because there are only 2 genders in this dataset, and there's no missing data.

    # 1 and 0 is enough to encode Sex

    X_train, X_test = labelEncode(X_train, X_test, "Sex")

    

    # for obtaining the median of age, try add more precision to titles

    X_train["Title"] = X_train["Name"].apply(getTitleFromNameBigOnly)

    X_test["Title"] = X_test["Name"].apply(getTitleFromNameBigOnly)



    medianPerTitleMap = medianPerTitle(X_train)

    def smarterMedianProbablyIdk(row):

        if np.isnan(row.Age):

            return medianPerTitleMap[row.Title]

        else:

            return row.Age

    X_train.loc[:,["Age"]] = X_train.apply(smarterMedianProbablyIdk, axis=1)

    X_test.loc[:,["Age"]] = X_test.apply(smarterMedianProbablyIdk, axis=1)

    X_train, X_test = stdScale(X_train, X_test, "Age")



    # create new feature, Family Size 

    def mxs(row):

        return row.SibSp + row.Parch

    X_train["FamilySize"] = X_train.apply(mxs, axis=1)

    X_test["FamilySize"] = X_test.apply(mxs, axis=1)

    X_train, X_test = stdScale(X_train, X_test, "FamilySize")



    X_train, X_test = stdScale(X_train, X_test, "SibSp")

    X_train, X_test = stdScale(X_train, X_test, "Parch")



    # Fare only has 1 missing data, filling with smart median is not worth it

    medianFare = X_train["Fare"].median()

    X_train.loc[:,["Fare"]] = X_train["Fare"].fillna(medianFare)

    X_test.loc[:,["Fare"]] = X_test["Fare"].fillna(medianFare)

    X_train, X_test = stdScale(X_train, X_test, "Fare")



    X_train = X_train.drop(["Cabin"], axis=1)

    X_test = X_test.drop(["Cabin"], axis=1)



    # OHE embarked, title

    modeEmbarked = X_train["Embarked"].mode()

    X_train.loc[:,["Embarked"]] = X_train["Embarked"].fillna(modeEmbarked)

    dum = ["Embarked", "Title"]

    X_train = X_train.join(pd.get_dummies(X_train[dum], prefix=dum))

    X_test = X_test.join(pd.get_dummies(X_test[dum], prefix=dum))



    X_train = X_train.drop(["Name", "Embarked", "Ticket", "Title"], axis=1)

    X_test = X_test.drop(["Name", "Embarked", "Ticket", "Title"], axis=1)



    return X_train, X_test
from sklearn.svm import SVC
def trainSVC(X_train, y_train, X_test):

    model = SVC(gamma='scale', C=1.0, degree=3, kernel='rbf')

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred



def trainML(X_train, y_train, X_test):

    return trainSVC(X_train, y_train, X_test)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
def rfattempt1(df_train, df_forSubs):

    kf = KFold(n_splits=5)



    oriFeatures = df_train.columns.drop(["PassengerId", "Survived"])

    X = df_train[oriFeatures].copy()

    y = df_train["Survived"].copy()



    aucList = []

    accList = []

    f1sList = []



    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X.iloc[train_idx,:].copy(), X.iloc[test_idx,:].copy()

        y_train, y_test = y[train_idx].copy(), y[test_idx].copy()



        X_train, X_test = rfattempt1_transform(X_train, X_test, y_train, y_test)



        model, y_pred = trainML(X_train, y_train, X_test)



        aucList.append(roc_auc_score(y_test, y_pred))

        accList.append(accuracy_score(y_test, y_pred))

        f1sList.append(f1_score(y_test, y_pred))

        

    print("Avg auc: {}".format(np.mean(aucList)))

    print("Avg acc: {}".format(np.mean(accList)))

    print("Avg F1-score: {}".format(np.mean(f1sList)))



    print("Predicting result in test.csv")

    X_p = df_forSubs[oriFeatures].copy()



    X_train, X_test = rfattempt1_transform(X, X_p, y)



    model, y_p = trainML(X_train, y, X_test)



    df_forSubs["Survived"] = y_p

    df_forSubs[["PassengerId", "Survived"]].to_csv("test.predicted.csv", index=False)



    print("Done, output in /kaggle/working/test.predicted.csv")
# reset df_train and df_test from input, because it has been modified previously

df_test = pd.read_csv("../input/titanic/test.csv")

df_train = pd.read_csv("../input/titanic/train.csv")

rfattempt1(df_train, df_test)