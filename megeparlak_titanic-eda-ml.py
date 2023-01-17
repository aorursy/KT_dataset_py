# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

from collections import Counter

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test["PassengerId"]
train.head()
train.info()
test.head()
test.info()
train["Sex"].value_counts().index.values
def barplot(a):

    b = train[a]

    c = b.value_counts()

    

    plt.figure(figsize=(10,5))

    plt.bar(c.index,c)

    plt.xticks(c.index, c.index.values)

    plt.title(a)

    plt.show()
category = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]

for i in category:

    barplot(i)
def histogram(a):

    plt.figure(figsize=(10,5))

    plt.hist(train[a], bins=100)

    plt.xlabel(a)

    plt.title(a)

    plt.show()
catnum2 = ["Fare", "Age","PassengerId"]

for i in catnum2:

    histogram(i)
plt.figure(figsize=(10,5))

sns.countplot(train.Pclass, hue=train.Survived);
plt.figure(figsize=(10,5))

sns.countplot(train.Sex, hue=train.Survived);
plt.figure(figsize=(10,5))

sns.countplot(train.SibSp, hue=train.Survived);
plt.figure(figsize=(10,5))

sns.countplot(train.Parch, hue=train.Survived);
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        Q1 = np.percentile(df[c],25)

        Q3 = np.percentile(df[c],75)

        IQR = Q3 - Q1

        outlier_step = IQR * 1.5

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train.loc[detect_outliers(train,["Age","SibSp","Parch","Fare"])]

train=train.drop(detect_outliers(train,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_len=len(train)

train=pd.concat([train,test],axis=0).reset_index(drop=True)
train.isnull().sum()
train["Embarked"]=train["Embarked"].fillna("C")
train["Fare"]=np.mean(train[train["Pclass"]==3]["Fare"])
list1=["SibSp","Parch","Survived"]

sns.heatmap(train[list1].corr(),annot= True, fmt=".2f")

plt.show()
sns.factorplot(x = "SibSp", y = "Survived", data= train, kind = "bar", size=7)

plt.show()
sns.factorplot(x = "Parch", y = "Survived", data= train, kind = "bar", size=7)

plt.show()
sns.factorplot(x = "Pclass", y = "Survived", data= train, kind = "bar", size=7)

plt.show()
g=sns.FacetGrid(train, col="Survived",size=5)

g.map(plt.hist,"Age", bins=20)

plt.show()
g=sns.FacetGrid(train, col="Survived", row="Pclass", size=3)

g.map(plt.hist,"Age",bins=20)

g.add_legend()

plt.show()
index_nan_age = list(train["Age"][train["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train["Age"][((train["SibSp"] == train.iloc[i]["SibSp"]) &(train["Parch"] == train.iloc[i]["Parch"])& (train["Pclass"] == train.iloc[i]["Pclass"]))].median()

    age_med = train["Age"].median()

    if not np.isnan(age_pred):

        train["Age"].iloc[i] = age_pred

    else:

        train["Age"].iloc[i] = age_med
name = train["Name"]

train["Title"] = [i.split(".")[0].split(",")[1].strip() for i in name]
sns.countplot(x="Title", data = train)

plt.xticks(rotation = 90)

plt.show()
train["Title"] = train["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train["Title"]]
sns.countplot(x="Title", data = train)

plt.xticks(rotation = 90)

plt.show()
g = sns.factorplot(x = "Title", y = "Survived", data = train, kind = "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train.drop(labels = ["Name"], axis = 1, inplace = True)
train = pd.get_dummies(train,columns=["Title"])
train["Fsize"] = train["SibSp"] + train["Parch"] + 1

train["family_size"] = [1 if i < 3 else 0 for i in train["Fsize"]]
train = pd.get_dummies(train, columns= ["family_size"])

train = pd.get_dummies(train, columns=["Embarked"])
tickets = []

for i in list(train.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train["Ticket"] = tickets
train = pd.get_dummies(train, columns= ["Ticket"])
train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns= ["Pclass"])
train["Sex"] = train["Sex"].astype("category")

train = pd.get_dummies(train, columns= ["Sex"])
train.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
test = train[train_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = train[:train_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
reg = LogisticRegression()

reg.fit(X_train, y_train)

log_train = round(reg.score(X_train, y_train)*100,2) 

log_test = round(reg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(log_train))

print("Testing Accuracy: % {}".format(log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),SVC(random_state = random_state),RandomForestClassifier(random_state = random_state),LogisticRegression(random_state = random_state),KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),"max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],"gamma": [0.001, 0.01, 0.1, 1],"C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],"min_samples_split":[2,3,10],"min_samples_leaf":[1,3,10],"bootstrap":[False],"n_estimators":[100,300],"criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),"penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),"weights": ["uniform","distance"],"metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,svc_param_grid,rf_param_grid,logreg_param_grid,knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)