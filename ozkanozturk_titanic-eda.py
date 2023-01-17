# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")  # for avaliable styles --> plt.style.avaliable



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

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot (variable):

    """

        input : variable ex: "Sex"

        output : bar plot and value count

    """

    # get feature

    var = train_df[variable]

    # count number of variable

    varValue = var.value_counts()

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue, edgecolor = "black")

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}". format(variable, varValue))

category1 = ['Survived', 'Sex','Pclass','Embarked','SibSp','Parch']

for each in category1:

    bar_plot(each)
category2 = ['PassengerId','Name','Age','Ticket','Cabin']

for each in category2:

    print("{}:\n".format(train_df[each].value_counts()))
def plot_hist(variable):

    plt.figure(figsize =(9,3))

    plt.hist(train_df[variable], bins =50, edgecolor = "black")

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("Frequency of {}".format(variable))

    plt.show()
numVariables = ["Fare", "Age", "PassengerId"]

for each in numVariables:

    plot_hist(each)
# Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
# Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending=False)
# Sibsp vs Survived

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
# ParCh vs Survived

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending = False)
# Single trial before creating function 



a = "SibSp"

q1 = np.percentile(train_df[[a]], 25)

q3 = np.percentile(train_df[[a]], 75)

iqr = q3 - q1

outlier_step = iqr*1.5

outliers = (train_df[a]<q1-outlier_step)|(train_df[a]>q3 + outlier_step)

outlier_indices = train_df[outliers].index

print("outliers:", q1-outlier_step, q3 + outlier_step)

print(type(outlier_indices))

outlier_indices

a = Counter(outlier_indices)

multiple_indices = [i for i, i in a.items() if i > 1]
def detect_outliers(df,features_of_df):

    

    outlier_indices_list=[]

    

    for each in features_of_df:

        # first quartile Q1

        Q1 = np.percentile(df[[each]], 25)

        # third quartile Q3

        Q3 = np.percentile(df[[each]], 75)

        # IQR = Q3 - Q1

        IQR = Q3 - Q1

        # outlier step = IQR x 1.5

        outlier_step = IQR*1.5

        # outliers = Q1 - outlier step or Q3 + outlier_step 

        outliers = (df[each] < Q1 - outlier_step) |(df[each]>Q3 + outlier_step) 

        # detect indeces of outliers in features of df

        outlier_indices= df[outliers].index

        # store indices

        outlier_indices_list.extend(outlier_indices)

    outlier_indices = Counter(outlier_indices_list)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    #print("multiple_outliers:", multiple_outliers, "\n\nOutlier_indices_list:",outlier_indices_list ) 

    return multiple_outliers

    
features = ["Age", "SibSp","Parch", "Fare"]

detect_outliers(train_df, features)
# Outliers in data 

train_df.loc[detect_outliers(train_df, features)]
# Drop outliers

train_df = train_df.drop(detect_outliers(train_df, features), axis = 0).reset_index(drop = True)

train_df.info()
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True) # If drop = True old index has been dropped, if False old index keeped as a column
train_df.info()
train_df.isnull()  # shows True--> if cell is null and False--> if cell is not null, type--> dataframe

train_df.isnull().any() # True/False: shows False that columns has no True value,type--> series / any()--> outputs True if there is atleast one True (not null or not zero) in column

train_df.columns[train_df.isnull().any()]   # Columns that has at least one null value
train_df.isnull().sum()  # Shows sum of nulls in each columns
# Embarked has 2 missing values

# Let us find these missing values

train_df[train_df.Embarked.isnull()]
f,ax = plt.subplots(figsize=(5,5))  # creating figure and axis

train_df.boxplot(column=["Fare"], by = "Embarked", ax = ax)

plt.show()
train_df["Embarked"].fillna(value="C", inplace = True)

#train_df.Embarked.isnull().sum()

#train_df[train_df.Embarked.isnull()]

train_df[train_df["Embarked"].isnull()]  # Check if value is placed.
# Find Missing data in Fare

train_df[train_df.Fare.isnull()]
train_df[train_df.Pclass == 3]  # Passengers having Pclass = 3
train_df[train_df.Pclass == 3].Fare.mean() # Mean of fare of passengers having Pclass = 3
# Missing Fare value to be filled with mean 12.7

train_df.Fare.fillna(value=(np.mean(train_df[train_df.Pclass==3].Fare)), inplace = True)

# Check

train_df[train_df.Fare.isnull()]
list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]

sns.heatmap(train_df[list1].corr(),annot = True, fmt=".2f")

plt.show()
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size =5)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x ="Parch", y = "Survived", data = train_df, kind = "bar", size = 5)

g.set_ylabels("Survived Propability")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived",data=train_df, kind = "bar", size = 5)

g.set_ylabels("Survived Probobality")

plt.show()
g = sns.FacetGrid(train_df, col = "Survived")  # creating different grids for each diff data in column survived of train_df

g.map(sns.distplot, "Age", bins=25) # mapping on g with func of sns.distplot and iterable "Age"

plt.show()
g = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=2 ) # creating grid with 2 columns(features 0&1) and 3 rows(features 1,2,3)

g.map(plt.hist, "Age", bins=25)

g.add_legend()

plt.show()
g= sns.FacetGrid(train_df, row = "Embarked", size=2)

g.map(sns.pointplot,"Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g= sns.FacetGrid(train_df, col ="Survived", row="Embarked", size = 2)

g.map(sns.barplot, "Sex", "Fare")

plt.show()
train_df[train_df.Parch==2].sort_values(by = "Name", ascending=True)
train_df[train_df.Age.isnull()]
# Age estimation according to Sex

sns.factorplot(x = "Sex", y = "Age", data=train_df, kind="box", size = 3)

plt.show()
# Age estimation according to Pclass

#sns.factorplot(data=train_df, x = "Pclass", y = "Age", kind="box", size=5)

sns.factorplot(x = "Sex", y = "Age", hue="Pclass", data=train_df, kind="box", size = 5)

plt.show()
# Age estimation according to Parch - SibSp 

sns.factorplot(x = "Parch", y = "Age",data=train_df, kind="box", size = 3)

sns.factorplot(x = "SibSp", y = "Age",data=train_df, kind="box", size = 3)

plt.show()
train_df.Sex = [1 if each=="male" else 0 for each in train_df.Sex]
sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True )

plt.show()
# Estimating missin Age value

# finding index of missing age values

index_nan_age = list(train_df[train_df.Age.isnull()].index) # or list(train_df["Age"][train_df["Age"].isnull()].index)



for each in  index_nan_age:

    # calculating median values of SibSp of each age missing values

    age_pred = train_df.Age[((train_df.SibSp == train_df.iloc[each].SibSp)&(train_df.Parch == train_df.iloc[each].Parch)&(train_df.Pclass == train_df.iloc[each].Pclass))].median()

    age_median = train_df.Age.median()

    if not np.isnan(age_pred):

        train_df.Age.iloc[each] = age_pred

    else:

        train_df.Age.iloc[each] = age_median



             
# As an example to understanding:

#i=1298

#train_df.Age[((train_df.SibSp == train_df.iloc[i].SibSp)&(train_df.Parch == train_df.iloc[i].Parch)&(train_df.Pclass == train_df.iloc[i].Pclass))].median

# Check if Age has nan values:

train_df[train_df.Age.isnull()]
train_df.info()
train_df.Name.head(10)
# Example of a single cell

train_df.Name[0]  # 'Braund, Mr. Owen Harris'

train_df.Name[0].split(".")[0] # 'Braund, Mr'

train_df.Name[0].split(".")[0].split(",")[-1] # ' Mr'

train_df.Name[0].split(".")[0].split(",")[-1].strip() # 'Mr'
# Extracting titles from Name and creating new column

train_df["Title"]= [each.split(".")[0].split(",")[-1].strip() for each in train_df.Name]

train_df["Title"].head(10)
# Distribution of Title in data

plt.subplots(figsize=(10,5))

sns.countplot(data=train_df, x = "Title")

plt.xticks(rotation = 45)

plt.show()
list(train_df.Title.value_counts().index)
# Convert all titles (Mr, Mrs, Mss, Dr ..etc ) into 4 categorical title

# ['Rev', 'Dr', 'Col', 'Major', 'Ms', 'Don', 'Jonkheer', 'Capt', 'Lady', 'the Countess', 'Sir', 'Dona']

train_df["Title"] = train_df.Title.replace(['Rev', 'Dr', 'Col', 'Major', 'Don', 'Jonkheer', 'Capt', 'Lady', 'the Countess', 'Sir', 'Dona'], "other")
# Master --> 0

# Miss, Ms, Mlle, Mrs --> 1

# Mr --> 2 

# other --> 3



train_df["Title"] = [0 if i=="Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i=="Mrs" else 2 if i=="Mr" else 3 for i in train_df.Title]
# Distribution of Title in data

#plt.subplots(figsize=(10,6))

sns.countplot(data=train_df, x = "Title")

plt.xticks(rotation = 0)

plt.show()
# Analysis of Titles vs Survival

g = sns.factorplot(data = train_df, x= "Title", y = "Survived", kind="bar")

g.set_xticklabels(["Master", "Mrs","Mr","Other"])

plt.show()
train_df.columns
# Drop column Name from data, because we have Title extracted from it.

train_df.drop(columns={"Name"}, inplace=True)  # or train_df.drop(label=["Name"], axis=1, inplace=True)

train_df.head()
# Converting Title into categorical(4 categories) ---> Title_0,Title_1,Title_2,Title_3

train_df = pd.get_dummies(train_df, columns=["Title"])

train_df.head()
# Converting Parch and SibSp into one column Family size

train_df["Fsize"] = train_df["Parch"] + train_df["SibSp"]+1

train_df.head()
g = sns.factorplot(x="Fsize", y="Survived", data=train_df, kind="bar")

train_df["family_size"] = [1 if each < 5 else 0 for each in train_df.Fsize]

train_df.head(10)
# Counts of Family size

sns.countplot(x="family_size", data = train_df)

plt.show()

print("total:", train_df.family_size.value_counts().values.sum(),"\n",train_df.family_size.value_counts())
# Family size survival rate

g = sns.factorplot(data = train_df, x = "family_size", y = "Survived", kind="bar")

g.set_ylabels("Survival rate")

plt.show()
# Categorising family_size

train_df = pd.get_dummies(data = train_df, columns=["family_size"])

train_df.head()

train_df.Embarked.head()
sns.countplot(x="Embarked", data=train_df)

plt.show()
train_df = pd.get_dummies(data = train_df, columns=["Embarked"])

train_df.head()
train_df.Ticket.head(20)
tickets = []

for each in list(train_df.Ticket):

    if not each.isdigit():

        tickets.append(each.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")     
train_df["Ticket"]=tickets

train_df.Ticket.value_counts()
train_df.head()
train_df= pd.get_dummies(train_df, columns=["Ticket"], prefix="T")

train_df.head()
sns.countplot(x = "Pclass", data = train_df)

plt.show()
train_df ["Pclass"] = train_df.Pclass.astype("category")  # converting to type category

train_df = pd.get_dummies(train_df, columns=["Pclass"])

train_df.head()
train_df["Sex"] = train_df.Sex.astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()
train_df.drop(["PassengerId", "Cabin"], axis = 1, inplace=True)

train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:] # split [from train_df_len to end]

test.drop(["Survived"], axis = 1, inplace = True)

test.head()
train = train_df[:train_df_len] # split [from start to train_df_len]

X_train = train.drop(["Survived"], axis = 1)

y_train = train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)



print("X_train len:", len(X_train))

print("y_train len:", len(y_train))

print("X_test len:", len(X_test))

print("y_test len:", len(y_test))

print("test len:", len(test))   # very first test split
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_logreg_train = round(logreg.score(X_train,y_train)*100,2)

acc_logreg_test = round(logreg.score(X_test, y_test)*100,2)



print("accuracy with train data", acc_logreg_train)

print("accuracy with test data", acc_logreg_test)
# grid to be created from parameters, for each algorithms, in a dictionary



# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree%20classifier#sklearn.tree.DecisionTreeClassifier



# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html



# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html



#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest%20classifier#sklearn.ensemble.RandomForestClassifier



# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



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

    
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, 

                          "ML Models":["DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means","ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_ylabel("Models")

plt.show()
best_estimators 
votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),("rf", best_estimators[2]),("lr", best_estimators[3])], voting = "soft", n_jobs = -1)



votingC.fit(X_train, y_train)





print("Accuracy score:",votingC.score(X_test, y_test))

test
test_survived = votingC.predict(test)

print("type test_survived",type(test_survived))

print("shape test_survived",np.shape(test_survived))

test_survived_series = pd.Series(test_survived, name="Survived").astype(int)  # Converting series ir order to concatenate

print("type test_survived_series",type(test_survived_series))

print("shape test_survived_series",np.shape(test_survived_series))



results = pd.concat([test_PassengerId, test_survived_series], axis=1)

results.to_csv("titanic_csv", index=False)