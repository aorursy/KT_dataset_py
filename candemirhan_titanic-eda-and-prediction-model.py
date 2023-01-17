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
train_df.head()
train_df.describe()
train_df.info()
train_df.columns
def baR_plot(variable):

    

    var = train_df[variable]

    varValue = var.value_counts()

    

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} \n {}".format(variable,varValue))
categoricVar = ["Survived", "Pclass","Name", "Sex" ,"SibSp", "Parch", "Ticket"]

for cat in categoricVar:

    baR_plot(cat)
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=800)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with Histogram".format(variable))

    plt.show
numericVar = ["Fare","Age","PassengerId"]

for num in numericVar:

    plot_hist(num)
# Pclass versus Survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# Sex versus Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# SibSp versus Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# Parch versus Survived

train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
def detectOutlier(df,features):

    

    outlier_indeces = list()

    for out in features:

        

        # 1st Quartile

        Q1 = np.percentile(df[out],25)

        # 3rd Quartile

        Q3 = np.percentile(df[out],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier Step

        outlier_step = IQR * 1.5

        # Outlier Detection and Indices

        outlier_list_col = df[(df[out] < Q1 - outlier_step) | (df[out] > Q3 + outlier_step)].index

        # Store Inedces

        outlier_indeces.extend(outlier_list_col)

    

    outlier_indeces = Counter(outlier_indeces)

    multiple_outliers = list(i for i,v in outlier_indeces.items() if v>2)

    

    return multiple_outliers
train_df.loc[detectOutlier(train_df,["Age","SibSp","Parch","Fare"])]
# Drop Outliers

train_df.drop(detectOutlier(train_df,["Age","SibSp","Parch","Fare"]),axis=0,inplace=True)
train_df_len = len(train_df)

df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

df.tail()
df.columns[df.isnull().any()]
df.isnull().sum()
df[df["Embarked"].isnull()]
df.boxplot(column="Fare",by="Embarked")

plt.show()
# B class Cabin



df[(df["Cabin"].str.startswith("B",na=False))]
# Fare of Cabines with error of 20 unit money



df[df["Fare"].between(60,100,inclusive=False)]
# Fare of B class Cabins with error of 20 unit money



df[(df["Cabin"].str.startswith("B",na=False)) & (df["Fare"].between(60,100,inclusive=False))]
# So, we can conclude that we can fill nan values with  Cherbourg



df["Embarked"].fillna("C",inplace=True)

df[df["Embarked"].isnull()]
df[df["Fare"].isnull()]
df[(df["Pclass"] == 3) & (df["Embarked"] == "S")]["Fare"]
df["Fare"].fillna(np.mean(df[(df["Pclass"] == 3) & (df["Embarked"] == "S")]["Fare"]),inplace=True)
df[df["Fare"].isnull()]
feature_list = ["SibSp","Parch","Age","Fare","Survived"]

plt.figure(figsize=(10,8))

sns.heatmap(df[feature_list].corr(),

           annot = True,

           fmt = ".2f"

           )

plt.show()
g = sns.factorplot(x = "SibSp", y = "Survived",

                   data = df,

                   kind = "bar",

                   size = 6

                   )

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived",

                kind = "bar",

                data = df,

                size = 6

                )

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived",

                  data = df,

                  kind = "bar",size = 6

                  )

g.set_ylabels("Survived Probability")

plt.show()
g = sns.FacetGrid(df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
g = sns.FacetGrid(df, col = "Survived", row = "Pclass", size = 2)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(df, row = "Embarked", size = 2)

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(df, row = "Embarked", col = "Survived", size = 2)

g.map(sns.barplot, "Sex","Fare")

g.add_legend()

plt.show()
df[df["Age"].isnull()]
sns.factorplot(x = "Sex", y = "Age",

               data = df,

               kind = "box"

              )

plt.show()
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",

              data = df,

              kind = "box"

              )

plt.show()
sns.factorplot(x = "Parch", y = "Age",

              data = df, 

              kind = "box"

              )

sns.factorplot(x = "SibSp", y = "Age",

              data = df,

              kind = "box"

             )

plt.show()
# Values of the Sex are in type of string. So we have to make it binary values to see the correlation with other features



df["Sex"] = [1 if i == "male" else 0 for i in df["Sex"]]

df.head()
sns.heatmap(df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True)

plt.show()
index_nan_age = list(df["Age"][df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = df["Age"][((df["SibSp"] == df.iloc[i]["SibSp"]) &

                         (df["Parch"] == df.iloc[i]["Parch"]) &

                         (df["Pclass"] == df.iloc[i]["Pclass"]))].median()

    age_med = df["Age"].median()

    if not np.isnan(age_pred):

        df["Age"].iloc[i] = age_pred

    else:

        df["Age"].iloc[i] = age_med
df[df["Age"].isnull()]
df["Name"].head(10)
name = df["Name"]

df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
df["Title"].head(10)
sns.countplot(x = "Title", data= df)

plt.xticks(rotation = 60)

plt.show()
# Convert to Categorical



df["Title"] = df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonk","Jonkheer","Dona"],"Other")

df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in df["Title"]]
g = sns.factorplot(x = "Title", y = "Survived",

                  data = df,

                  kind = "bar"

                  )

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
df.drop(labels=["Name"], axis = 1, inplace = True)
df.head()
df = pd.get_dummies(df, columns = ["Title"])

df.head()
df.head()
df["FSize"] = df["SibSp"] + df["Parch"] + 1
df.head()
g = sns.factorplot(x = "FSize", y = "Survived",

                  data = df,

                  kind = "bar"

                  )

g.set_ylabels("Survival Probability")

plt.show()
df["Family_Size"] = [1 if i < 5 else 0 for i in df["FSize"]]
df.head(10)
sns.countplot(x = "Family_Size", data = df)

plt.show()
g = sns.factorplot(x = "Family_Size", y = "Survived",

                  data = df,

                  kind = "bar"

                  )

g.set_ylabels("Survival Probability")

plt.show()
df = pd.get_dummies(df,columns = ["Family_Size"])

df.head()
df["Embarked"]
sns.countplot(x = "Embarked", data = df)

plt.show()
df = pd.get_dummies(df, columns = ["Embarked"])
df.head()
df["Ticket"].head(20)
a = "W./C. 14258//"

a.replace(".","").replace("/","").strip().split()[0]
tickets = list()

for i in list(df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".", "").replace("/", "").strip().split()[0])

    else:

        tickets.append("x")

df["Ticket"] = tickets
df.Ticket.head(20)
df.head(10)
df = pd.get_dummies(df, columns=["Ticket"], prefix = "T")
df.head(10)
sns.countplot(x = "Pclass", data = df)

plt.show()
df["Pclass"] = df["Pclass"].astype("category")
df = pd.get_dummies(df,columns = ["Pclass"])

df.head(20)
df["Sex"] = df["Sex"].astype("category")

df = pd.get_dummies(df, columns = ["Sex"])

df.head()
df.drop(labels = ["PassengerId","Cabin"],axis = 1, inplace=True)
df.columns
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
train_df_len
test = df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
test.head()
train = df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

acc_logreg_train = round(logreg.score(X_train,y_train)*100,2)

acc_logreg_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy : {} %".format(acc_logreg_train))

print("Testing Accuracy : {} %".format(acc_logreg_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth" : range(1,20,2)

                }

svc_param_grid = {"kernel" : ["rbf"],

                 "gamma" : [0.001, 0.01, 0.1, 1],

                 "C" : [1, 10, 50, 100, 200, 300, 1000]

                 }

rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]

                }

logreg_param_grid = {"C" : np.logspace(-3,3,7),

                    "penalty" : ["l1","l2"]

                    }

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]

                 }

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid

                   ]

cv_result = list()

best_estimator = list()

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv= StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1, verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimator.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means": cv_result,

                           "ML Models": ["DecisionTreeClassifier", "SVM","RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier"]

                          })

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimator[0]),

                                         ("rf",best_estimator[2]),

                                         ("lr",best_estimator[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train,y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

result = pd.concat([test_PassengerId, test_survived], axis = 1)

result.to_csv("titanic.csv",index=False)