# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Plotting libraries:

import seaborn as sbn

import matplotlib.pyplot as plt



# Encoding and Transformation libraries:

from sklearn.preprocessing import OneHotEncoder



# ML libraries

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_init_df=pd.read_csv('/kaggle/input/titanic/train.csv')

train_init_df.head()
test_init_df=pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_init_df["PassengerId"]

test_init_df.head()
train_init_df.describe()
train_init_df.isnull().sum()
print(len(train_init_df))
train_init_df=train_init_df.drop(['Cabin','PassengerId'],axis=1)
train_init_df[train_init_df['Age'].isnull()].head(10)
g = sbn.FacetGrid(train_init_df, col = "Survived")

g.map(sbn.distplot, "Age", bins = 25)

plt.show()
g = sbn.FacetGrid(train_init_df, col = "Survived", row = "Pclass", height = 5)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()
train_init_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df[train_init_df['Name'].isnull()].head(10)
def checkNameSalutations(name):

    salutation = re.findall(r'(\w{2,})\.',name)

    return salutation[0]
train_init_df['Salutation']=train_init_df['Name'].apply(checkNameSalutations)

train_init_df.head()
train_init_df[["Salutation","Survived"]].groupby(["Salutation"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def binaryGender(val):

    if val=='male':

        return 0

    elif val=='female':

        return 1
train_init_df['Sex']= train_init_df['Sex'].apply(binaryGender)

train_init_df.head()
train_init_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def checkAplphabetInTicket(ticket):

    alpha=re.findall(r'(\w{1,}\D)',ticket)

    if len(alpha)!=0:

        return 1

    else:

        return 0
train_init_df['Alpha_ticket']= train_init_df['Ticket'].apply(checkAplphabetInTicket)

train_init_df.head()
train_init_df['Alpha_ticket'].value_counts()
train_init_df[["Alpha_ticket","Survived"]].groupby(["Alpha_ticket"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_init_df["Salutation"] = train_init_df["Salutation"].replace(["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_init_df["Salutation"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_init_df["Salutation"]]

train_init_df["Salutation"].head(20)
sbn.countplot(x="Salutation", data = train_init_df)

plt.xticks(rotation = 60)

plt.show()
g = sbn.catplot(x  = "Salutation", y = "Survived", data  = train_init_df, kind = "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_init_df.drop(labels= ["Name"], axis = 1, inplace = True)

train_init_df.head()
train_init_df.drop(labels= ["Ticket"], axis = 1, inplace = True)
train_init_df.isnull().sum()
train_init_df["Embarked"] = train_init_df["Embarked"].fillna("C")

train_init_df[train_init_df["Embarked"].isnull()]
index_nan_age = list(train_init_df["Age"][train_init_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_init_df["Age"][((train_init_df["SibSp"] == train_init_df.iloc[i]["SibSp"]) &(train_init_df["Parch"] == train_init_df.iloc[i]["Parch"])& (train_init_df["Pclass"] == train_init_df.iloc[i]["Pclass"]))].median()

    age_med = train_init_df["Age"].median()

    if not np.isnan(age_pred):

        train_init_df["Age"].iloc[i] = age_pred

    else:

        train_init_df["Age"].iloc[i] = age_med
train_init_df.head()
train_init_df = pd.get_dummies(train_init_df,columns = ["Embarked"])

train_init_df.head()
train_init_df = pd.get_dummies(train_init_df,columns = ["Salutation"])

train_init_df.head()
train_init_df = pd.get_dummies(train_init_df,columns = ["Pclass"])

train_init_df.head()
# train_init_df.drop(labels= ["Survived"], axis = 1, inplace = True)
train_init_df = pd.get_dummies(train_init_df,columns = ["Alpha_ticket"])

train_init_df.head()
train_init_df["Sex"] = train_init_df["Sex"].astype("category")

train_init_df = pd.get_dummies(train_init_df, columns = ["Sex"])

train_init_df.head()
train_init_df["Fsize"] = train_init_df["SibSp"] + train_init_df["Parch"] + 1

train_init_df.head()
train_init_df["family_size"] = [1 if i < 5 else 0 for i in train_init_df["Fsize"]]

train_init_df.head()
train_init_df = pd.get_dummies(train_init_df, columns= ["family_size"])

train_init_df.head()
test_df = train_init_df.copy(deep=True)

test_df.drop(labels = ["Survived"],axis = 1, inplace  = True)
test_df.head()
train_init_df.head()
train = train_init_df.copy(deep=True)

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test_df))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2)

acc_log_test = round(logreg.score(X_test, y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sbn.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_init_df.head()
test_init_df.drop(['Cabin'],axis=1,inplace=True)
test_final_sub=test_init_df['PassengerId']
test_final_sub.head()
test_init_df['Salutation']=test_init_df['Name'].apply(checkNameSalutations)

test_init_df.head()
test_init_df["Salutation"] = test_init_df["Salutation"].replace(["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

test_init_df["Salutation"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in test_init_df["Salutation"]]

test_init_df["Salutation"].head(20)
test_init_df.head()
test_init_df.isnull().sum()
test_init_df.drop(['Name'],axis=1,inplace=True)
test_init_df['Sex']= test_init_df['Sex'].apply(binaryGender)

test_init_df.head()
test_init_df['Alpha_ticket']= test_init_df['Ticket'].apply(checkAplphabetInTicket)

test_init_df.head()
test_init_df['Alpha_ticket'].value_counts()
test_init_df.drop(labels= ["Ticket"], axis = 1, inplace = True)

test_init_df.head()
index_nan_age = list(test_init_df["Age"][test_init_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = test_init_df["Age"][((test_init_df["SibSp"] == test_init_df.iloc[i]["SibSp"]) &(test_init_df["Parch"] == test_init_df.iloc[i]["Parch"])& (test_init_df["Pclass"] == test_init_df.iloc[i]["Pclass"]))].median()

    age_med = test_init_df["Age"].median()

    if not np.isnan(age_pred):

        test_init_df["Age"].iloc[i] = age_pred

    else:

        test_init_df["Age"].iloc[i] = age_med
test_init_df[test_init_df["Fare"].isnull()]
test_init_df["Fare"] = test_init_df["Fare"].fillna(np.mean(test_init_df[test_init_df["Pclass"] == 3]["Fare"]))
test_init_df[test_init_df["Fare"].isnull()]
test_init_df = pd.get_dummies(test_init_df,columns = ["Embarked"])

test_init_df.head()
test_init_df = pd.get_dummies(test_init_df,columns = ["Salutation"])

test_init_df.head()
test_init_df = pd.get_dummies(test_init_df,columns = ["Pclass"])

test_init_df.head()
test_init_df = pd.get_dummies(test_init_df,columns = ["Alpha_ticket"])

test_init_df.head()
test_init_df["Sex"] = test_init_df["Sex"].astype("category")

test_init_df = pd.get_dummies(test_init_df, columns = ["Sex"])

test_init_df.head()
test_init_df["Fsize"] = test_init_df["SibSp"] + test_init_df["Parch"] + 1

test_init_df.head()
test_init_df["family_size"] = [1 if i < 5 else 0 for i in test_init_df["Fsize"]]

test_init_df.head()
test_init_df = pd.get_dummies(test_init_df, columns= ["family_size"])

test_init_df.head()
train_init_df.head()
test_init_df.drop(['PassengerId'],axis=1,inplace=True)

test_init_df.head()
test_survived = pd.Series(votingC.predict(test_init_df), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)
# test_survived = pd.Series(votingC.predict(test_df), name = "Survived").astype(int)

# results = pd.concat([test_PassengerId, test_survived],axis = 1)

# results.to_csv("titanic.csv", index = False)