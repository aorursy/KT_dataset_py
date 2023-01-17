# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid") # adds plots some grids.



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
train_titanic_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_titanic_data = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
type(train_titanic_data)
train_titanic_data.head()
train_titanic_data.columns
train_titanic_data.info()
train_titanic_data.describe()
train_titanic_data["Sex"]
train_titanic_data.Sex
train_titanic_data.Sex.count
train_titanic_data.Sex.value_counts()
train_titanic_data.Sex.value_counts().index
train_titanic_data.Sex.value_counts().values
def Bar_Plot(variable):

    plt.figure(figsize = (9,3))

    plt.bar(train_titanic_data[variable].value_counts().index, train_titanic_data[variable].value_counts())

    plt.ylabel("Frequency")

    plt.title(str(variable))

    plt.show()

    print("{}: \n{}".format(variable, train_titanic_data[variable].value_counts()))

    

variables = ["Survived", "Sex", "Pclass","Embarked","SibSp", "Parch"]

for variable in variables:

    Bar_Plot(variable)
other_variables = ["Cabin", "Name", "Ticket"]

for variable in other_variables:

    print("{}: \n".format(train_titanic_data[variable].value_counts()))
def Histogram_Plot(variable):

    plt.figure(figsize = (9,3))

    train_titanic_data[variable].plot(kind = "hist", color = "green", bins = 80)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(variable + " distribution with hist")

    plt.show()
Histogram_Plot("Fare")
Histogram_Plot("Age")
Histogram_Plot("PassengerId")
train_titanic_data[["Pclass", "Survived"]]
train_titanic_data[["Pclass", "Survived"]].groupby(train_titanic_data["Pclass"], as_index = False).mean()
train_titanic_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean()
train_titanic_data[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean()
train_titanic_data[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean()
# Q1 = np.percentile(data[variable],25)

# Q3 = np.percentile(data[variable],75)

# IQR = Q3 - Q1

# outlier_list = data[(data[variable] < IQR * 1.5) | (data[variable] > IQR * 1.5)].index       
train_titanic_data.boxplot(column = "Pclass", by = "Sex")

plt.show()
train_df_len = len(train_titanic_data)

train_titanic_data = pd.concat([train_titanic_data,test_titanic_data],axis = 0).reset_index(drop = True)
train_df_len
len(train_titanic_data)
train_titanic_data.head()
whole_titanik_data = pd.concat([train_titanic_data,test_titanic_data], axis = 0)
whole_titanik_data
whole_titanik_data.columns
whole_titanik_data.columns[whole_titanik_data.isnull().any()]
whole_titanik_data.isnull().sum()
whole_titanik_data[whole_titanik_data["Embarked"].isnull()]
whole_titanik_data.boxplot(column = "Fare", by = "Embarked")

plt.show()
whole_titanik_data["Embarked"] = whole_titanik_data["Embarked"].fillna("C")
whole_titanik_data[whole_titanik_data["Embarked"].isnull()]
whole_titanik_data[whole_titanik_data["Fare"].isnull()]
whole_titanik_data["Fare"] = whole_titanik_data["Fare"].fillna(np.mean(whole_titanik_data[whole_titanik_data["Pclass"] == 3]["Fare"]))
whole_titanik_data[whole_titanik_data["Fare"].isnull()]
plt.figure(figsize = (10,10))

data_list = train_titanic_data[["SibSp", "Parch", "Age", "Fare", "Survived"]]

sns.heatmap(data_list.corr(), annot = True, linewidths = .5, fmt = ".2f")

plt.title("Correlation Between SibSp, Parch, Age, Fare and Survived")

plt.show()
sns.factorplot(x = "SibSp", y = "Survived", data = train_titanic_data, kind = "bar", palette = "muted")

plt.ylabel("Survived Probability")

plt.show()
sns.factorplot(x = "Parch", y = "Survived", data = train_titanic_data, kind = "bar", palette = "muted")

plt.ylabel("Survived Probability")

plt.show()
sns.factorplot(x = "Pclass", y = "Survived", data = train_titanic_data, kind = "bar", palette = "muted")

plt.ylabel("Survived Probability")

plt.show()
graph = sns.FacetGrid(train_titanic_data, col = "Survived")

graph.map(sns.distplot, "Age", bins = 25)

plt.show()
g = sns.FacetGrid(train_titanic_data, col = "Survived", row = "Pclass", size = 3)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()
graph = sns.FacetGrid(train_titanic_data, col = "Embarked")

graph.map(sns.pointplot, "Pclass", "Survived", "Sex", bins = 25)

plt.legend()

plt.show()
g = sns.FacetGrid(train_titanic_data, row = "Embarked", col = "Survived", size = 3)

g.map(sns.barplot, "Sex", "Fare")

plt.legend()

plt.show()
train_titanic_data[train_titanic_data.Age.isnull()]
sns.boxplot(x = "Sex", y = "Age", data = train_titanic_data)

plt.show()
sns.boxplot(x = "Sex", y = "Age", hue = "Pclass", data = train_titanic_data)

plt.show()
sns.boxplot(x = "Parch", y = "Age", data = train_titanic_data)

plt.show()
sns.boxplot(x = "SibSp", y = "Age", data = train_titanic_data)

plt.show()
sns.heatmap(train_titanic_data[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)

plt.show()
train_titanic_data["Sex"] = [0 if i == "female" else 1 for i in train_titanic_data.Sex]
train_titanic_data["Sex"]
sns.heatmap(train_titanic_data[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)

plt.show()
index_nan_age = list(train_titanic_data["Age"][train_titanic_data["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_titanic_data["Age"][((train_titanic_data["SibSp"] == train_titanic_data.iloc[i]["SibSp"]) &(train_titanic_data["Parch"] == train_titanic_data.iloc[i]["Parch"])& (train_titanic_data["Pclass"] == train_titanic_data.iloc[i]["Pclass"]))].median()

    age_med = train_titanic_data["Age"].median()

    if not np.isnan(age_pred):

        train_titanic_data["Age"].iloc[i] = age_pred

    else:

        train_titanic_data["Age"].iloc[i] = age_med
train_titanic_data[train_titanic_data.Age.isnull()]
train_titanic_data.Name.head(10)
names = train_titanic_data.Name

names
name_list = []

for name in names:

    name = name.split(".")  

    name_list.append(name)
name_list[0:5]
first_part = []

for words in name_list:

    first_part.append(words[0])

    
first_part[0:5]
gender_list = []

for word in first_part:

    word = word.split(",")

    gender_list.append(word)
gender_list[0:5]
genders = []

for gender in gender_list:

    genders.append(gender[-1].strip())
genders[0:5]
train_titanic_data["Title"] = genders
train_titanic_data["Title"].head(10)
plt.figure(figsize = (10,6))

sns.countplot(x = "Title", data = train_titanic_data)

plt.xticks(rotation = 60)

plt.show()
train_titanic_data["Title"] = train_titanic_data["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_titanic_data["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_titanic_data["Title"]]
train_titanic_data["Title"].head(10)
sns.countplot(x = train_titanic_data.Title)

plt.xticks(rotation = 60)

plt.show()
sns.factorplot(x = "Title", y = "Survived", kind = "bar",data = train_titanic_data, palette = "muted")

plt.xlabel("Title")

plt.ylabel("Survived Probability")

plt.show()
train_titanic_data = train_titanic_data.drop("Name", axis = 1)
train_titanic_data.head()
train_titanic_data["Title"] = train_titanic_data["Title"].astype("category")

train_titanic_data = pd.get_dummies(train_titanic_data, columns= ["Title"])

train_titanic_data.head()
train_titanic_data.head()
train_titanic_data["Family_Size"] = train_titanic_data.SibSp + train_titanic_data.Parch + 1
train_titanic_data.head()
sns.factorplot(x = "Family_Size", y = "Survived", data = train_titanic_data, kind = "bar")

plt.xlabel("Family Size")

plt.ylabel("Survived")

plt.show()
train_titanic_data["Survived_Family"] = [1 if i < 5 else 0 for i in train_titanic_data["Family_Size"]]
train_titanic_data.head()
sns.countplot(x = "Survived_Family", data = train_titanic_data)

plt.show()
sns.factorplot(x = "Survived_Family", y = "Survived", data = train_titanic_data, kind = "bar")

plt.show()
train_titanic_data.Embarked.head()
sns.countplot(x = "Embarked", data = train_titanic_data)

plt.show()
train_titanic_data["Embarked"] = train_titanic_data["Embarked"].astype("category")

train_titanic_data = pd.get_dummies(train_titanic_data, columns= ["Embarked"])

train_titanic_data.head()
train_titanic_data.Ticket.head()
tickets = []

for i in list(train_titanic_data.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_titanic_data["Ticket"] = tickets
train_titanic_data.head(20)
train_titanic_data["Ticket"] = train_titanic_data["Ticket"].astype("category")

train_titanic_data = pd.get_dummies(train_titanic_data, columns= ["Ticket"])

train_titanic_data.head()
sns.countplot(x = "Pclass", data = train_titanic_data)

plt.show()
train_titanic_data["Pclass"] = train_titanic_data["Pclass"].astype("category")

train_titanic_data = pd.get_dummies(train_titanic_data, columns= ["Pclass"])

train_titanic_data.head()
train_titanic_data["Sex"] = train_titanic_data["Sex"].astype("category")

train_titanic_data = pd.get_dummies(train_titanic_data, columns= ["Sex"])

train_titanic_data.head()
train_titanic_data.columns
train_titanic_data = train_titanic_data.drop(["Cabin"], axis = 1)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_titanic_data[train_df_len:]

test.drop(labels = ["Survived"], axis = 1, inplace = True)
test.head()
train_titanic_data.head()
train = train_titanic_data[:train_df_len]

x_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)
print("x_train",len(x_train))

print("x_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(x_train, y_train)
accuracy_logreg_train = logreg.score(x_train, y_train)

print("Training accuracy: %", round(accuracy_logreg_train*100,2))
accuracy_logreg_test = logreg.score(x_test, y_test)

print("Testing accuracy: %", round(accuracy_logreg_test*100,2))
classifier = [DecisionTreeClassifier(random_state = 42), SVC(random_state = 42), RandomForestClassifier(random_state = 42),

              LogisticRegression(random_state = 42), KNeighborsClassifier()]





dt_param_grid = {"min_samples_split" : range(10,500,20), "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"], "gamma": [0.001, 0.01, 0.1, 1], "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10], "min_samples_split":[2,3,10], "min_samples_leaf":[1,3,10],

                "bootstrap":[False], "n_estimators":[100,300], "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7), "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(), "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



classifier_param = [dt_param_grid, svc_param_grid, rf_param_grid, logreg_param_grid, knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]), ("rfc",best_estimators[2]), ("lr",best_estimators[3])], voting = "soft", n_jobs = -1)



votingC = votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test),y_test)*100)