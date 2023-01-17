# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.info()
def bar_plot(variable):

    """

    input: Variable, ex: "Sex"

    output: Bar plot & Variable count

    

    """

    #get feature

    var = train_df[variable]

    #count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable], bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with Histogram".format(variable))

    plt.show()
NumericVar = ["Fare","Age", "PassengerId"]

for c in NumericVar:

    plot_hist(c)
# Pclass - Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending= False)
# Sex - Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index= False).mean().sort_values(by= "Survived",ascending=False)
# SibSp - Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by = "Survived", ascending=False)
# Parch - Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(df,features):

    outlier_indices=[]

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # Detect outliers and their indices

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # Store indices

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i , v in outlier_indices.items() if v>2)

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df, ["Age", "SibSp","Parch","Fare"])]
# Drop outliers

train_df = train_df.drop(detect_outliers(train_df, ["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df.loc[detect_outliers(train_df, ["Age", "SibSp","Parch","Fare"])]
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare", by = "Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
train_df[train_df["Fare"].isnull()]
train_df
list1=["SibSp", "Parch", "Age", "Fare", "Pclass", "Survived"]

f, ax=plt.subplots(figsize=(11,9))

sns.heatmap(train_df[list1].corr(), annot=True, fmt=" .2f", ax=ax)

plt.show()
g=sns.factorplot(x="SibSp", y="Survived", data=train_df, size=6, kind="bar")

g.set_ylabels("Possibility of Survival")

plt.show()
g=sns.factorplot(data=train_df, x="Parch", y="Survived", kind="bar", size=6)

g.set_ylabels("Possibility of Survival")

plt.show()
g=sns.factorplot(data=train_df, kind="bar", x="Pclass", y="Survived", size=6)

g.set_ylabels("Possibility of Survival")

plt.show()
g=sns.FacetGrid(train_df, col="Survived", height=5)

g.map(sns.distplot, "Age", bins=25)

plt.show()
g=sns.FacetGrid(train_df, col="Survived", row="Pclass", size=3)

g.map(plt.hist, "Age", bins=25)

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df, "Embarked", size=3)

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g=sns.FacetGrid(train_df, col="Survived", row="Embarked", size=3)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
train_df[train_df.Age.isnull()]
sns.factorplot(data=train_df, x="Sex", y="Age", kind="box")

plt.show()
sns.factorplot(data=train_df, x="Pclass", y="Age", hue="Sex", kind="box")

plt.show()
sns.factorplot(data=train_df, x="Parch", y="Age", kind="box")

sns.factorplot(data=train_df, x="SibSp", y="Age", kind="box")

plt.show()
train_df["Sex01"]=[1 if i=="male" else 0 for i in train_df.Sex]
sns.heatmap(train_df[["Age", "Sex01", "Pclass", "SibSp", "Parch"]].corr(), annot=True)

plt.show()
index_nan_age=list(train_df[train_df.Age.isnull()].index)

for i in index_nan_age:

    age_pred=train_df.Age[((train_df.SibSp==train_df.iloc[i].SibSp)&(train_df.Pclass==train_df.iloc[i].Pclass)&(train_df.Parch==train_df.iloc[i].Parch))].median()

    age_med=train_df.Age.median()

    if not np.isnan(age_pred):

        train_df.Age.iloc[i]=age_pred

    else:

        train_df.Age.iloc[i]=age_med
train_df[train_df.Age.isnull()]
train_df.Name
Title=[i.split(".")[0].split(",")[-1].strip() for i in train_df.Name]

train_df["Title"]=Title
f,ax=plt.subplots(figsize=(18,7))

sns.countplot(x=train_df.Title)

plt.show()
train_df.Title=train_df.Title.replace(["Don", "Rev", "Dr", "Mme", "Major", "Lady", "Sir", "Col", "Capt", "the Countess", "Jonkheer", "Dona"], "Other")
train_df.Title.unique()
new_title=[]

for i in train_df.Title:

    if i=="Master":

        new_title.append(0)

    elif i=="Miss" or i=="Mrs" or i=="Ms" or i=="Mlle":

        new_title.append(1)

    elif i=="Mr":

        new_title.append(2)

    elif i=="Other":

        new_title.append(3)

set(new_title)
train_df.Title=new_title
f,ax=plt.subplots(figsize=(18,7))

sns.countplot(x=train_df.Title)

plt.show()
g=sns.factorplot(x="Title", y="Survived", data=train_df, kind="bar")

g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])

g.set_ylabels("Survival Possibility")

plt.show()
train_df=pd.get_dummies(train_df, columns=["Title"])
train_df.head()
train_df["Fsize"]=train_df.Parch+train_df.SibSp+1
g=sns.factorplot(x="Fsize", y="Survived", data=train_df, kind="bar")

g.set_ylabels("Survival Possibility")

plt.show()
new_fsize=[]

for each in train_df.Fsize:

    if each<=4:

        new_fsize.append(1)

    elif each>4:

        new_fsize.append(0)

train_df["family_size"]=new_fsize
train_df
g=sns.factorplot(x="family_size", y="Survived", data=train_df, kind="bar")

g.set_ylabels("Survival Possibility")

plt.show()
train_df=pd.get_dummies(train_df, columns=["family_size"])
train_df=pd.get_dummies(train_df, columns=["Embarked"])
train_df.Ticket
new_ticket=[]

for each in train_df.Ticket:

    if not each.isdigit():

        new_ticket.append(each.replace(".", "").replace("/", "").strip().split(" ")[0])

    else:

        new_ticket.append("x")

train_df["Ticket"]=new_ticket
train_df=pd.get_dummies(train_df, columns=["Ticket"], prefix="T")
sns.countplot(x="Pclass", data=train_df)

plt.show()
train_df=pd.get_dummies(train_df, columns=["Pclass"])
train_df=pd.get_dummies(train_df, columns=["Sex"])
train_df.drop("Sex01", axis=1, inplace=True)
train_df
train_df.drop(["PassengerId", "Cabin"], axis=1, inplace=True)
train_df
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
train_data=train_df[:train_df_len]
len(train_data)
test_data=train_df[train_df_len:]
len(test_data)
x=train_data.drop(["Survived", "Name", "SibSp", "Parch"], axis=1)

y=train_data["Survived"]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)
x_train
logreg=LogisticRegression()

logreg.fit(x_train, y_train)

train_acc=round(logreg.score(x_train, y_train)*100,3)

test_acc=round(logreg.score(x_test, y_test)*100,3)

print("Training Accuracy:", train_acc, "%")

print("Test Accuracy:", test_acc, "%")
rs=42

classifier=[DecisionTreeClassifier(random_state=rs),

           SVC(random_state=rs),

           RandomForestClassifier(random_state=rs),

           KNeighborsClassifier(),

           LogisticRegression(random_state=rs)]



dt_param_grid={"min_samples_split": range(10,500,20), 

               "max_depth": range(1,20,2)}



svc_param_grid={"kernel": ["rbf"], 

               "gamma": [0.001, 0.01, 0.1, 1],

               "C": [1,10,50,100,200,300,1000]}



rf_param_grid={"max_features": [1,3,10],

              "min_samples_split": [2,3,10],

              "min_samples_leaf": [1,3,10],

              "bootstrap": [False],

              "n_estimators": [100,300],

              "criterion": ["gini"]}



knn_param_grid={"n_neighbors": np.linspace(1,19,10, dtype=int).tolist(),

               "weights": ["distance", "uniform"],

               "metric": ["euclidean", "manhattan"]}



logreg_param_grid={"C": np.logspace(-3,3,7),

                  "penalty": ["l1", "l2"]}



classifier_param=[dt_param_grid, svc_param_grid, rf_param_grid, knn_param_grid, logreg_param_grid]
cv_result=[]

best_estimators=[]

for i in range(len(classifier)):

    clf=GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1, verbose=1)

    clf.fit(x_train, y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_result=[100*each for each in cv_result]
results=pd.DataFrame({"Cross Validation Best Scores": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression"]})

f,ax=plt.subplots(figsize=(12,7))

g = sns.barplot(data=results, y="ML Models", x="Cross Validation Best Scores")

g.set_ylabel("")

g.set_xlabel("Accuracy %")

plt.show()

for i in range(len(results)):

    print(results["ML Models"][i], "Accuracy:", results["Cross Validation Best Scores"][i], "%")
voting_c=VotingClassifier(estimators=[("dt", best_estimators[0]), ("rf", best_estimators[2]), ("lr", best_estimators[4])],

                         voting="soft", n_jobs=-1)

voting_c=voting_c.fit(x_train, y_train)

print("Accuracy:", 100*accuracy_score(voting_c.predict(x_test), y_test), "%")
test=test_data.drop(["Survived", "Name", "Parch", "SibSp"], axis=1)

test_survived=pd.Series(voting_c.predict(test), name="Survived").astype(int)

results=pd.concat([test_PassengerId, test_survived], axis=1)
results.to_csv("submission.csv", index=False)