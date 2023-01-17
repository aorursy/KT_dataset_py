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

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    """

        input : variable ex: "Sex"

        output : bar plot & value count

    """

    #getting feature

    var = train_df[variable]

    

    #count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    #visualize

    

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))

    
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp","Parch"]

for c in category1:

    bar_plot(c)
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubiton with histogram".format(variable))

    plt.show()

    
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
#Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)
#Sex vs Survived

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived",ascending = False)
#SibSp vs Survived

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived",ascending = False)
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived",ascending = False)
def det_outlier(df,features):

    outlier_indices = []

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3-Q1 

        #Outlier step

        outlier_step = IQR * 1.5

        

        #detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q1 + outlier_step)].index

        

        # store indeces

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    

    multiple_outliers = list(i for i, v in outlier_indices.items() if v >2) #study it

    

    return multiple_outliers
train_df.loc[det_outlier(train_df, ["Age","SibSp","Parch","Fare"])]

# drop outliers

train_df = train_df.drop(det_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

train_df
#fitting train and test data frames

train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis= 0).reset_index(drop = True)
train_df.head()
#finding missing values

train_df.columns[train_df.isnull().any()]



#finding count missing values

train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
#firstly embarked

train_df.boxplot(column = "Fare", by = "Embarked")

plt.show()
#I looked the ports and saw C port has most likely right for these 2 person.

train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
#now fare

np.mean(train_df[train_df["Pclass"] == 3]["Fare"])

#filling with the mean value.

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

train_df[train_df["Fare"].isnull()]
#seaborn library used

#corr method is correlation

#annot = True --> you can see numbers.

list1 = ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot = True, fmt = ".2f")

plt.show()
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df ,kind = "bar", size = 6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x= "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)

g = sns.FacetGrid(train_df, col = "Survived")

g.map(sns.distplot, "Age", bins = 25)

plt.show()
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)

g.map(plt.hist,"Age", bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", size = 3)

g.map(sns.pointplot,"Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 3)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
#trying with sex

sns.factorplot(x = "Sex", y = "Age", data= train_df, kind = "box")

plt.show()
#trying with Pclass

sns.factorplot(x = "Sex", y = "Age",hue = "Pclass", data= train_df, kind = "box")

plt.show()
#trying with parch and sibsp

sns.factorplot(x = "Parch", y = "Age", data= train_df, kind = "box")

sns.factorplot(x = "SibSp", y = "Age", data= train_df, kind = "box")

plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot = True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    age_median = train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else:

        train_df["Age"].iloc[i] = age_median
train_df[train_df["Age"].isnull()]
name = train_df["Name"]

#split before and after "." then take first one [0] then split before and after "," then take title "mr" or "mrs"

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)
sns.countplot(x= "Title", data= train_df)

plt.xticks(rotation = 60)

plt.show()
#convert to categorical

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i== "Mr" else 3 for i in train_df["Title"]]
sns.countplot(x= "Title", data= train_df)

plt.xticks(rotation = 60)

plt.show()
g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels = ["Name"] , axis = 1 ,inplace = True)
train_df.head(5)
train_df = pd.get_dummies(train_df, columns=["Title"])
train_df.head(10)
#We will combine to parch and sibsp to new feature named Family Size

train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head(9)
#correlation between fsize with survived

g = sns.factorplot(x = "Fsize", y= "Survived", data = train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df["family_size"] = [1 if i <5 else 0 for i in train_df["Fsize"]]
train_df.head(10)
sns.countplot(x = "family_size", data = train_df)

plt.show()
g = sns.factorplot(x ="family_size", y = "Survived", data = train_df , kind =  "bar")

g.set_ylabels("Survived Probability")

plt.show()
train_df = pd.get_dummies(train_df, columns = ["family_size"])
train_df.head(15)
sns.countplot(x = "Embarked", data = train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns = ["Embarked"])
train_df.head(5)
train_df["Ticket"].head(20)
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")



train_df["Ticket"] = tickets
train_df = pd.get_dummies(train_df , columns = ["Ticket"],prefix = "T")

train_df.head(10)
sns.countplot(x = "Pclass", data = train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df , columns = ["Pclass"])

train_df.head(10)
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df , columns = ["Sex"])

train_df.head(10)
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)

train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"], axis = 1, inplace = True)

test.head()
train = train_df[:train_df_len]

X_train = train.drop(labels= "Survived", axis = 1)

y_train = train["Survived"]

X_train , X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.33,random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_test",len(y_test))

print("y_train",len(y_train))
print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train,y_train)* 100,2)

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: {}".format(acc_log_train))

print("Testing Accuracy: {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

              SVC(random_state = random_state),

              RandomForestClassifier(random_state = random_state),

              LogisticRegression(random_state = random_state),

              KNeighborsClassifier()

             ]
dt_param_grid = {"min_samples_split": range(10,500,20),

                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel": ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf": [1,3,10],

                "bootstrap": [False],

                "n_estimators": [100,300],

                "criterion": ["gini"]}

logreg_param_grid = {"C": np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype= int).tolist(),

                 "weights":["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                    svc_param_grid,

                    rf_param_grid,

                    logreg_param_grid,

                    knn_param_grid]
cv_results = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1, verbose = 1)

    clf.fit(X_train,y_train)

    cv_results.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_results[i])

    
cv_result_bar = pd.DataFrame({"Cross Validation Means": cv_results, "ML Models": ["DecisionTreeClassifier", "SVM",

                                                                                 "RandomForestClassifier",

                                                                                 "LogisticRegression",

                                                                                 "KNeighborsClassifier"]})

g= sns.barplot("Cross Validation Means", "ML Models", data = cv_result_bar)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")

plt.show()
votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),

                                        ("rfc", best_estimators[2]),

                                        ("lr", best_estimators[3])],

                                        voting = "soft", n_jobs = -1)



#voting = "hard" is making simpler operation.

votingC = votingC.fit(X_train,y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId,test_survived], axis = 1)

results.to_csv("titanic.csv", index = False)