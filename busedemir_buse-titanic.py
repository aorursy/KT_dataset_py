# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# linear algebra

import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

#plt.style.use("seaborn-whitegrid")

import matplotlib.pyplot as plt

# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

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
plt.style.available
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test_df["PassengerId"]
train_df.info
train_df.columns
train_df.head()
train_df.describe()
total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_df.info()
def bar_plot(variable):

    """

        input: variable ex:"Sex"

        output: bar plot & value count

    """

    #get feature

    var = train_df[variable]

    #count number of categorical variable (value/sample)

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable, varValue))
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar=["Fare", "Age", "PassengerId"]

for n in numericVar:

    plot_hist(n)
#Average survival of those sitting in 1., 2., 3. class

# Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending=False)
# Pclass vs Sex

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending=False)
# Pclass vs SibSp

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending=False)
# Pclass vs Parch

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending=False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3 - Q1

        #Outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers

    
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#drop outliers

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
#In order not to lose size

train_df_len = len(train_df)

#We combined data frames

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
#Is there a #null value or not?

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
#It is not clear where 2 passengers got on the ship

#We can compare according to something material feature.

train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by = "Embarked")

plt.show()

#1. the most paid ones got on ship from C station.

#Lowest payers got on ship from Q station.
train_df["Embarked"]= train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
#I have assigned the average of 3rd class passengers.

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]
#Correlation between the features that I have studied

list1 = ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot = True, fmt= ".2f")

plt.show()

#Fare feature seems to have correlation with survived feature (0.26)
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()

#Having a lot of SipSp have less chance to survive

#if SibSp==0 or 1 or 2, passenger has more chance to survive

#We can consider a new feature describing these categories
g=sns.factorplot(x="Parch", y="Survived", kind="bar", data=train_df, size=6)

g.set_ylabels("Survived Probability")

plt.show()

#Small families have a higher chance of survival than large families and individuals.
g=sns.factorplot(x="Pclass", y="Survived", data=train_df, kind="bar", size=6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()

#Age<=10 has a high survival rate

#oldest passengers(80) survived,

#large number of 20 years old did not survive

#most passengers are in 15-35 age range

#use age feature in training

#use age distribution for missing value of age
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)

g.map(plt.hist, "Age", bins = 25)

g.add_legend()

plt.show()

#The rate of living in first class passengers is higher than the others; The rate of living in third class passengers is lower than the others.

#Pclass is important feature for model train
g = sns.FacetGrid(train_df, row = "Embarked", size = 3)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()

#Female passenger much better survival rate than males

#Males have better survival rate in pclass 3 in C

#Embarked and Sex will be used in training
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 3)

g.map(sns.barplot, "Sex","Fare")

g.add_legend()

plt.show()

#Passengers who pay higher fare have better survival
train_df[train_df["Age"].isnull()]
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")

plt.show()

#Sex is not informative for age prediction, age distribution seems to be same
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = train_df, kind = "box")

plt.show()

#1st class passengers are older than 2sd, and 2nd is older than 3rd class
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")

sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")

plt.show()
train_df["Sex"]=[1 if i=="male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)

plt.show()

#Age is not correleted with sex but it is correleted with Parch, SibSp and Pclass
#Finding the empty ones and indexes of age properties

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

#Browse these indexes and predict Age by looking at SibSp, Parch and Pclass.

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & 

                                (train_df["Parch"] == train_df.iloc[i]["Parch"]) & 

                                (train_df["Pclass"] == train_df.iloc[i]["Parch"]))].median()

#Use median for cases where we can't predict.

    age_median = train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else:

        train_df["Age"].iloc[i] = age_median
train_df[train_df["Age"].isnull()]
train_df["Name"].head(10)
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].unique()
sns.countplot(x= "Title",data = train_df)

plt.xticks(rotation = 60)

plt.show()
#Convert to categorical

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess",

                                               "Capt","Col","Don",

                                               "Dr","Major","Rev","Sir",

                                               "Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" 

                     or i == "Ms" or i == "Mlle" or i == "Mrs"

                     else 2 if i == "Mr" else 3 for i in train_df["Title"]]
sns.countplot(x= "Title",data = train_df)

plt.xticks(rotation = 60)

plt.show()
g=sns.factorplot(x="Title", y="Survived", data=train_df, kind="bar")

g.set_xticklabels(["Master", "Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
#I removed the Name column because I created the Title column.

train_df.drop(labels = ["Name"],axis  =1, inplace = True)
train_df.head
train_df = pd.get_dummies(train_df,columns = ["Title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1 

train_df.head()
g = sns.factorplot(x  ="Fsize",y = "Survived",data = train_df,kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(20)
sns.countplot(x="family_size", data=train_df)

plt.show()
g = sns.factorplot(x  ="family_size",y = "Survived",data = train_df,kind = "bar")

g.set_ylabels("Survival")

plt.show()

#Small families have more change to survive than large families
train_df=pd.get_dummies(train_df, columns=["family_size"])

train_df.head()
sns.countplot(x="Embarked", data=train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head()
train_df["Ticket"].head(20)
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df["Ticket"].head(20)
train_df.head()
train_df = pd.get_dummies(train_df, columns=["Ticket"], prefix="T")

train_df.head(10)
sns.countplot(x = "Pclass",data = train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df,columns = ["Sex"])

train_df.head(10)
train_df.drop(labels = ["PassengerId","Cabin"],axis = 1,inplace = True)
train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
test = train_df[train_df_len:]

#There won't be survived column in the test.

test.drop(labels = ["Survived"],axis = 1, inplace = True)

test.head(10)
train = train_df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2)

acc_log_test = round(logreg.score(X_test, y_test)*100,2)

print("Training Accuracy:%{}".format(acc_log_train))

print("Testing Accuracy:%{}".format(acc_log_test))
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

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], 

                       cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result,

                           "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

#According to my votingC classifier, I predict X_test and then compare y_test to accuracy skore.

print(accuracy_score(votingC.predict(X_test),y_test))

test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)
test_survived
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(votingC, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))
from sklearn.metrics import f1_score

f1_score(y_train, predictions)
from sklearn.metrics import precision_recall_curve



# getting the probabilities of our predictions

y_scores = votingC.predict_proba(X_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
def plot_precision_vs_recall(precision, recall):

    plt.plot(recall, precision, "g--", linewidth=2.5)

    plt.ylabel("recall", fontsize=19)

    plt.xlabel("precision", fontsize=19)

    plt.axis([0, 1.5, 0, 1.5])



plt.figure(figsize=(14, 7))

plot_precision_vs_recall(precision, recall)

plt.show()
from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)