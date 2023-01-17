# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassenngerId = test_df["PassengerId"]
# kinda unnecessary when .columns is used

train_df.head()
train_df.columns
test_df.head()
train_df.describe()
test_df.describe()
train_df.info()
def bar_plot(variable):

    """

        input = variable, e.g. sex

        output = bar plot & value count

    """

    var = train_df[variable]

    varValue = var.value_counts()

    

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable, varValue))
cat1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in cat1:

    bar_plot(c)
cat2 = ["Cabin", "Name", "Ticket"]

for c in cat2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with histogram" .format(variable))

    plt.show
cat3 = ["Fare", "Age", "PassengerId"]

for c in cat3:

    plot_hist(c)
train_df[["Pclass", "Survived"]]
#pclass vs survived



train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
#sex and survived



train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
#sipsp and survived



train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
#parch vs survived



train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
def detectOutlier(df, features):

    outlierIndices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c], 25)

        # 3rd quartile

        Q3 = np.percentile(df[c], 75)

        #IQR

        IQR = Q3 - Q1

        #outlier step

        outlierStep = IQR * 1.5

        #detect outlier and indices

        outlierListCol = df[(df[c] < Q1 - outlierStep) | (df[c] > Q3 + outlierStep)].index

        #store indices

        outlierIndices.extend(outlierListCol)

        

    outlierIndices = Counter(outlierIndices) #counter shows how many of a single element exists

    multipleOutliers = list(i for i, v in outlierIndices.items() if v > 2)

    

    return multipleOutliers
train_df.loc[detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"])]
#dropping outliers

train_df = train_df.drop(detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare", by = "Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Fare"].isnull()]
train_df.boxplot(column = "Fare", by = "Pclass")

plt.show()
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
listo = ["SibSp", "Parch", "Age", "Fare", "Survived"]



sns.heatmap(train_df[listo].corr(), annot = True, fmt = ".2f")

plt.show()
g = sns.factorplot(x= "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("survived prob sibsp")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", size = 9)

g.set_ylabels("survived prob")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 5)

plt.show()
g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")

g.map(plt.hist, "Age", bins = 25)

plt.show()
g = sns.FacetGrid(train_df, row= "Embarked")

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")

plt.show()
sns.factorplot(x = "Sex", y = "Age", hue= "Pclass", data = train_df, kind = "box")

plt.show()
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")

sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")

plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)
indexes = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in indexes:

    ageP = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    ageM = train_df["Age"].median()

    if not np.isnan(ageP):

        train_df["Age"].iloc[i] = ageP

    else: 

        train_df["Age"].iloc[i] = ageM
train_df[train_df["Age"].isnull()]
train_df["Name"].head(10)
name = train_df["Name"]

train_df["title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
sns.countplot(x = "title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
train_df["title"] = train_df["title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "other")

train_df["title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["title"]]

train_df["title"].head(20)
sns.countplot(x = "title", data = train_df)

plt.xticks(rotation = 60)

plt.show()
g = sns.factorplot(x = "title", y = "Survived", data = train_df, kind = "bar")

g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Other"])

g.set_ylabels("Survival")

plt.show()
train_df.drop(columns = ["Name"], inplace = True)
train_df = pd.get_dummies(train_df, columns = ["title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["famsize"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df.head()
sns.countplot(x = "famsize", data = train_df)

plt.show()
g = sns.factorplot(x = "famsize", y = "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df = pd.get_dummies(train_df, columns = ["famsize"])
sns.countplot(x = "Embarked", data = train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()
train_df["Ticket"].head(30)
tickets = []

for i in (train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".", " ").replace("/", " ").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

train_df.head(10)
sns.countplot(x = "Pclass", data = train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
test.head()
train = train_df[:train_df_len]

x_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(x_train))

print("X_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(x_train, y_train)



acctrain = round(logreg.score(x_train, y_train) * 100, 2)

acctest = round(logreg.score(x_test, y_test) * 100, 2)

print("train data accuracy: %", acctrain)

print("test data accuracy: %", acctest)
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state), SVC(random_state = random_state),

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

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression", "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassenngerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)