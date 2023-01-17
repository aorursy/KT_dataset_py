import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

from collections import Counter
import os

import warnings
warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_passengerId=test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
variable = "Survived"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['red', 'green'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Survived",train_df["Survived"].value_counts()))

variable = "Sex"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['blue', 'red'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Sex",train_df["Survived"].value_counts()))
variable = "Pclass"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['red', 'green','blue'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Passenger Class",train_df["Pclass"].value_counts()))
variable = "Embarked"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['red', 'green','blue'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Embarked",train_df["Embarked"].value_counts()))
variable = "SibSp"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['red', 'green','blue','yellow','magenta','cyan'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Siblings/Spouses",train_df["SibSp"].value_counts()))
variable = "Parch"
columnvalues = train_df[variable]
columncounts = columnvalues.value_counts()
#plot graph
plt.figure(figsize=(9,3))
plt.bar(columncounts.index,columncounts, color=['red', 'green','blue','yellow','magenta','cyan'])
plt.xticks(columncounts.index,columncounts.index.values)
plt.ylabel("Frequency")
plt.title(variable)
plt.show()
print("{} \n {}".format("Parent/Children",train_df["Parch"].value_counts()))
#Cabin Variable
print('{} \n',format(train_df["Cabin"].value_counts()))
#Name Variable
print('{} \n',format(train_df["Name"].value_counts()))
#Name Variable
print('{} \n',format(train_df["Ticket"].value_counts()))
variable = "Fare"
plt.figure(figsize=(9,3))
plt.hist(train_df[variable],bins=10)
plt.xlabel(variable)
plt.ylabel("Frequency")
plt.title("{} distribution with Histogram".format(variable))
plt.show()
variable = "Age"
plt.figure(figsize=(9,3))
plt.hist(train_df[variable])
plt.xlabel(variable)
plt.ylabel("Frequency")
plt.title("{} distribution with Histogram".format(variable))
plt.show()
variable = "PassengerId"
plt.figure(figsize=(9,3))
plt.hist(train_df[variable],bins=891)
plt.xlabel(variable)
plt.ylabel("Frequency")
plt.title("{} distribution with Histogram".format(variable))
plt.show()
featurelist = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[featurelist].corr(), annot = True, cmap="RdYlGn")
plt.show()
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#Plot Survived Percentage
g = sns.catplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", height = 6)
g.set_ylabels("Percent of Survived")
plt.show()
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#print Survived Percentage
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#Plot Survived Percentage
g = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", height = 6)
g.set_ylabels("Percent of Survived")
plt.show()
#Print Survived Percentage
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#Plot Survived Percentage
g = sns.catplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", height = 6)
g.set_ylabels("Percent of Survived")
plt.show()
g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()
def find_outliers(dataframe,variables,threshold):
    outliers = []
    
    for c in variables:
        # First Part
        Quartile1 = np.percentile(dataframe[c],25)
        # Last Part
        Quartile3 = np.percentile(dataframe[c],75)
        # IQR
        IQR = Quartile3 - Quartile1
        outlier_step = IQR * 1.5
        # Find outlier indexes
        outlier_list_col = dataframe[(dataframe[c] < Quartile1 - outlier_step) | (dataframe[c] > Quartile3 + outlier_step)].index
        # store indexes
        outliers.extend(outlier_list_col)
    
    outliers = Counter(outliers)
    multiple_outliers = list(i for i, v in outliers.items() if v > threshold)
    
    return multiple_outliers
train_df.loc[find_outliers(train_df,["Age","SibSp","Parch","Fare"],1)]
train_df.loc[find_outliers(train_df,["Age","SibSp","Parch","Fare"],2)]
# drop 2 threshold outlier records 
train_df = train_df.drop(find_outliers(train_df,["Age","SibSp","Parch","Fare"],2),axis = 0).reset_index(drop = True)
#train dataframe missing value columns
train_df.columns[train_df.isnull().any()]
#missing values count in train data
train_df.isnull().sum()
#Find which passengers embarked ports are unknown...
train_df[train_df["Embarked"].isnull()]
#draw graph: Embarked Vs. Fare
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()
#Fill Missing Embarked Value with C
train_df["Embarked"] = train_df["Embarked"].fillna("C")
# Control filling
train_df[train_df["Embarked"].isnull()]
#Find which passengers don't have age data
train_df[train_df["Age"].isnull()]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True, cmap="RdYlGn")
plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med
train_df[train_df["Age"].isnull()]

#test dataframe missing value columns
test_df.columns[test_df.isnull().any()]
#missing values count in train data
test_df.isnull().sum()
#Find which passenger ticket price is Null
test_df[test_df["Fare"].isnull()]
#Average value of Pclass=3
np.mean(test_df[test_df["Pclass"] == 3]["Fare"])
test_df["Fare"] = test_df["Fare"].fillna(np.mean(test_df[test_df["Pclass"] == 3]["Fare"]))
test_df[test_df["Fare"].isnull()]
#Find which passengers don't have age data
test_df[test_df["Age"].isnull()]
sns.heatmap(test_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True, cmap="RdYlGn")
plt.show()
index_nan_age = list(test_df["Age"][test_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = test_df["Age"][((test_df["SibSp"] == test_df.iloc[i]["SibSp"]) &(test_df["Parch"] == test_df.iloc[i]["Parch"])& (test_df["Pclass"] == test_df.iloc[i]["Pclass"]))].median()
    age_med = test_df["Age"].median()
    column = test_df["Age"]
    if not np.isnan(age_pred):
        column.iloc[i] = age_pred
    else:
        column.iloc[i] = age_med
test_df[test_df["Age"].isnull()]
justrainlen = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df["Name"].head(10)
name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head()
g = sns.catplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Percentage of Survival")
plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()
train_df["tempFamilySize"] = train_df["SibSp"] + train_df["Parch"]
train_df.head()
g = sns.catplot(x = "tempFamilySize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Percentage of Survived")
plt.show()
train_df["familySize"] = [1 if i < 4 else 0 for i in train_df["tempFamilySize"]]
g = sns.catplot(x = "familySize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Percentage of Survived")
plt.show()
train_df.drop(labels = ["tempFamilySize"], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df, columns= ["familySize"])
train_df.head()
train_df["Ticket"].head(20)
tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("X")
train_df["Ticket"] = tickets
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
train_df.head(10)
#For embarked 
train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()
#for Pclass
train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df.head()
#For Sex
train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])
train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
train_df.columns
TEST = train_df[justrainlen:]
TEST.drop(labels = ["Survived"],axis = 1, inplace = True)
TEST.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
TRAIN = train_df[:justrainlen]
TRAINX = TRAIN.drop(labels = "Survived", axis = 1)
TRAINY = TRAIN["Survived"]
X_train, X_test, y_train, y_test = train_test_split(TRAINX, TRAINY, test_size = 0.30)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(TEST))
#for same results, fix the random value
random_state = 7
from sklearn.linear_model import LogisticRegression
#default settings
logreg = LogisticRegression(random_state=random_state)
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("DEFAULT:")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

#parameter penalty
logreg = LogisticRegression(random_state=random_state, penalty='none',C=0.1)
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("PARAMETERS:")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
from sklearn.tree import DecisionTreeClassifier
dectre = DecisionTreeClassifier(random_state=random_state)
dectre.fit(X_train, y_train)
acc_log_train = round(dectre.score(X_train, y_train)*100,2) 
acc_log_test = round(dectre.score(X_test,y_test)*100,2)
print("DEFAULT:")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

dectre = DecisionTreeClassifier(random_state=random_state, criterion="entropy", min_samples_split=0.5, max_depth=1.5)
dectre.fit(X_train, y_train)
acc_log_train = round(dectre.score(X_train, y_train)*100,2) 
acc_log_test = round(dectre.score(X_test,y_test)*100,2)
print("PARAMETERS:")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
acc_log_train = round(svc.score(X_train, y_train)*100,2) 
acc_log_test = round(svc.score(X_test,y_test)*100,2)
print("DEFAULT")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

svc = SVC(kernel="poly",degree=5)
svc.fit(X_train, y_train)
acc_log_train = round(svc.score(X_train, y_train)*100,2) 
acc_log_test = round(svc.score(X_test,y_test)*100,2)
print("PARAMETERS")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
from sklearn.ensemble import RandomForestClassifier
rndfor = RandomForestClassifier(random_state = random_state)
rndfor.fit(X_train, y_train)
acc_log_train = round(rndfor.score(X_train, y_train)*100,2) 
acc_log_test = round(rndfor.score(X_test,y_test)*100,2)
print("DEFAULT")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

rndfor = RandomForestClassifier(random_state = random_state,n_estimators=200,max_depth=5)
rndfor.fit(X_train, y_train)
acc_log_train = round(rndfor.score(X_train, y_train)*100,2) 
acc_log_test = round(rndfor.score(X_test,y_test)*100,2)
print("PARAMETERS")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_log_train = round(knn.score(X_train, y_train)*100,2) 
acc_log_test = round(knn.score(X_test,y_test)*100,2)
print("DEFAULT")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

knn = KNeighborsClassifier(n_neighbors=10,radius=2.0)
knn.fit(X_train, y_train)
acc_log_train = round(knn.score(X_train, y_train)*100,2) 
acc_log_test = round(knn.score(X_test,y_test)*100,2)
print("PARAMETERS")
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
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
    print(clf.best_estimator_)
    print(cv_result[i])
from sklearn.ensemble import VotingClassifier
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Accuracy")
g.set_title("Scores")
from sklearn.metrics import accuracy_score
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(TEST), name = "Survived").astype(int)

test_PassengerId = test_df["PassengerId"]
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)
test_survived = pd.Series(dectre.predict(TEST), name = "Survived").astype(int)
acc_log_train = round(dectre.score(X_train, y_train)*100,2) 
acc_log_test = round(dectre.score(X_test,y_test)*100,2)
test_PassengerId = test_df["PassengerId"]
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic-basic-dt.csv", index = False)