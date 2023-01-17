# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head(10)   # See first 10 rows
df_test.tail(10)   # See last 10 rows
df_train.info()   # Give informatin about features
df_train.isnull().sum()  # Check null values
df_train.describe()  #statistical informations
df_train.corr()  # Check colleration with features
categorical_features = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]

for each in categorical_features:
    plt.figure(figsize = (6,4))
    feature = df_train[each]
    count_variable = feature.value_counts()
    plt.bar(count_variable.index, count_variable)
    plt.ylabel("Total Count Variable")
    plt.title(each)
    plt.show()
nümerical_features = ["Fare", "Age"]

for each in nümerical_features:
    plt.figure(figsize = (6,4))
    feature = df_train[each]
    count_variable = feature.value_counts()
    plt.bar(count_variable.index, count_variable)
    plt.ylabel("Total Count Variable")
    plt.title(each)
    plt.show()
sns.distplot(df_train['Age']);
sns.distplot(df_train['Fare']);
sns.pairplot(df_train)    # See all plots amongst all features
fig, ax = plt.subplots(figsize=(12,7))
sns.heatmap(df_train.corr(), annot=True, linewidths = .5, ax = ax);
features = ["Age","SibSp","Parch","Fare"]
outlier_indices = []
for each in features:
    Q1 = np.percentile(df_train[each],25)    # 1st quartile
    Q3 = np.percentile(df_train[each],75)   # 3rd quartile
    IQR = Q3 - Q1                           # IQR
    outlier_step = IQR * 1.5                # outlier steps formula
    outlier_list_col = df_train[(df_train[each] < Q1 - outlier_step) | (df_train[each] > Q3 + outlier_step)].index  # detect outlier and their indeces
    outlier_indices.extend(outlier_list_col)
from collections import Counter

outlier_indices = Counter(outlier_indices)
multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
multiple_outliers    # These are outliers' indices
def detect_outliers(df,features):        # need dataframe and features
    outlier_indices = []
    
    for each in features:
        Q1 = np.percentile(df[each],25)
        Q3 = np.percentile(df[each],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[each] < Q1 - outlier_step) | (df[each] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
df_train.loc[detect_outliers(df_train,features)]
df_train = df_train.drop(detect_outliers(df_train,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
df_train.head(10)
df_train.isnull().sum()    # Again we should check null values
df_train[df_train["Embarked"].isnull()]
df_train["Embarked"] = df_train["Embarked"].fillna("C")   # Fill embarked feature
df_train[df_train["Embarked"].isnull()]   # Again check this feature
df_train[df_train["Age"].isnull()].head(10)    # Check another feature
sns.factorplot(x = "Parch", y = "Age", data = df_train, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = df_train, kind = "box")
plt.show()
index_nan_age = list(df_train["Age"][df_train["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = df_train["Age"][((df_train["SibSp"] == df_train.iloc[i]["SibSp"]) &(df_train["Parch"] == df_train.iloc[i]["Parch"])& (df_train["Pclass"] == df_train.iloc[i]["Pclass"]))].median()
    age_med = df_train["Age"].median()
    if not np.isnan(age_pred):
        df_train["Age"].iloc[i] = age_pred
    else:
        df_train["Age"].iloc[i] = age_med
df_train[df_train["Age"].isnull()]   # Check this feature
sex_variable = []
for each in df_train["Sex"]:
    if each == 'male':
        i = 1
        sex_variable.append(i)
    else:
        i = 0
        sex_variable.append(i)
        
df_train["Sex"] = sex_variable    
g = sns.factorplot(x = "SibSp", y = "Survived", data = df_train, kind = "bar", size = 5)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = df_train, size = 5)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.FacetGrid(df_train, col = "Survived")
g.map(sns.distplot, "Age", bins = 15)
plt.show()
g = sns.FacetGrid(df_train, col = "Survived", row = "Pclass", size = 4)
g.map(plt.hist, "Age", bins = 15)
g.add_legend()
plt.show()
df_train.head()
name = df_train["Name"]

pdlist = []

for i in name:
    name_change = i.split(".")[0].split(",")[-1].strip()
    pdlist.append(name_change)
    
df_train["Title"] = pdlist
df_train["Title"] = df_train["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

title_variables = []

for i in df_train["Title"]:
    if i == "Master":
        i = 0
    elif i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs":
        i == 1
    
    elif i == "Mr":
        i == 2 
    
    else:
        i == 3
        
df_train["Title"] = title_variables
df_train["Fsize"] = df_train["SibSp"] + df_train["Parch"] + 1

for i in df_train["Fsize"]:
    if i < 5:
        i = 1
    else:
        i = 0
sns.countplot(x = "family_size", data = df_train)
plt.show()
df_train.head(10)   # Check dataset
df_train = pd.get_dummies(df_train, columns= ["family_size"])
df_train.head()
sns.countplot(x = "Embarked", data = df_train)
plt.show()
df_train = pd.get_dummies(df_train, columns=["Embarked"])
df_train.head(10)  # Check dataset
tickets = []
for i in list(df_train.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
        
df_train["Ticket"] = tickets
df_train = pd.get_dummies(df_train, columns= ["Ticket"], prefix = "T")
df_train.head()
sns.countplot(x = "Pclass", data = df_train)
plt.show()
df_train["Pclass"] = df_train["Pclass"].astype("category")
df_train = pd.get_dummies(df_train, columns= ["Pclass"])

df_train["Sex"] = df_train["Sex"].astype("category")
df_train = pd.get_dummies(df_train, columns=["Sex"])
df_train.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
df_train.head(10)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1), 
                                                    df_train['Survived'], test_size=0.30, 
                                                    random_state=42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
from sklearn.linear_model import LogisticRegression
LR  = LogisticRegression()
LR.fit(X_train, y_train)

print("Train Datasets Accuracy: {}".format(LR.score(X_train, y_train)))
print("Test Datasets Accuracy: {}".format(LR.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 300,max_depth = 5,min_samples_split = 3)
RFC.fit(X_train,y_train)

print("Train Datasets Accuracy: {}".format(RFC.score(X_train,y_train)))
print("Test Datasets Accuracy: {}".format(RFC.score(X_test,y_test)))
RFC = RandomForestClassifier(n_estimators = 300 ,max_depth = 5, min_samples_split = 10)
RFC.fit(X_train,y_train)

print("Train Datasets Accuracy: {}".format(RFC.score(X_train,y_train)))
print("Test Datasets Accuracy: {}".format(RFC.score(X_test,y_test)))
RFC = RandomForestClassifier(n_estimators = 500 ,max_depth = 10 , min_samples_split = 6)
RFC.fit(X_train,y_train)

print("Train Datasets Accuracy: {}".format(RFC.score(X_train,y_train)))
print("Test Datasets Accuracy: {}".format(RFC.score(X_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 30  # as an example

knn = KNeighborsClassifier(n_neighbors = n_neighbors) # n_neighbors = k

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)
print(" {} nn score: {} ".format(n_neighbors, knn.score(X_test,y_test)))
score_list = []

for each in range(10,40):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test,y_test))
    
plt.plot(range(10,40),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

print("accuracy of decision tree: ", dt.score(X_test,y_test))
from lightgbm import LGBMClassifier

LGB = LGBMClassifier(learning_rate = 0.01,max_depth = 5, num_leaves = 3)
LGB.fit(X_train,y_train)

print("Train Datasets Accuracy: {}".format(LGB.score(X_train,y_train)))
print("Test Datasets Accuracy: {}".format(LGB.score(X_test,y_test)))
