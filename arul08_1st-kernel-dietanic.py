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
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df.head()
train_df.describe()
print(pd.isnull(train_df).sum()) 
print(pd.isnull(test_df).sum()) 
train_df.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

test_df.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
train_df1 = train_df.copy()

train_df1.dropna(inplace = True)

sns.distplot(train_df1["Age"])
#the median will be an acceptable value to place in the NaN cells

train_df["Age"].fillna(train_df["Age"].median(), inplace = True)

test_df["Age"].fillna(test_df["Age"].median(), inplace = True) 

train_df["Embarked"].fillna("S", inplace = True)

test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)



print(pd.isnull(train_df).sum())

print(" ")

print(pd.isnull(test_df).sum()) 
copy = train_df.copy()

copy.dropna(inplace = True)

sns.distplot(copy["Age"])
#can ignore the testing set for now

sns.barplot(x="Sex", y="Survived", data=train_df)

plt.title("Distribution of Survival based on Gender")

plt.show()



total_survived_females = train_df[train_df.Sex == "female"]["Survived"].sum()

total_survived_males = train_df[train_df.Sex == "male"]["Survived"].sum()



print("Total people survived is: " + str((total_survived_females + total_survived_males)))

print("Proportion of Females who survived:") 

print(total_survived_females/(total_survived_females + total_survived_males))

print("Proportion of Males who survived:")

print(total_survived_males/(total_survived_females + total_survived_males))

sns.barplot(x="Pclass", y="Survived", data=train_df)

plt.ylabel("Survival Rate")

plt.title("Distribution of Survival Based on Class")

plt.show()



total_survived_one = train_df[train_df.Pclass == 1]["Survived"].sum()

total_survived_two = train_df[train_df.Pclass == 2]["Survived"].sum()

total_survived_three = train_df[train_df.Pclass == 3]["Survived"].sum()

total_survived_class = total_survived_one + total_survived_two + total_survived_three



print("Total people survived is: " + str(total_survived_class))

print("Proportion of Class 1 Passengers who survived:") 

print(total_survived_one/total_survived_class)

print("Proportion of Class 2 Passengers who survived:")

print(total_survived_two/total_survived_class)

print("Proportion of Class 3 Passengers who survived:")

print(total_survived_three/total_survived_class)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_df)

plt.ylabel("Survival Rate")

plt.title("Survival Rates Based on Gender and Class")
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_df)

plt.ylabel("Survival Rate")

plt.title("Survival Rates Based on Gender and Class")
survived_ages = train_df[train_df.Survived == 1]["Age"]

not_survived_ages = train_df[train_df.Survived == 0]["Age"]

plt.subplot(1, 2, 1)

sns.distplot(survived_ages, kde=False)

plt.axis([0, 100, 0, 100])

plt.title("Survived")

plt.ylabel("Proportion")

plt.subplot(1, 2, 2)

sns.distplot(not_survived_ages, kde=False)

plt.axis([0, 100, 0, 100])

plt.title("Didn't Survive")

plt.show()
sns.stripplot(x="Survived", y="Age", data=train_df, jitter=True)
sns.pairplot(train_df)
train_df.sample(5)
test_df.sample(5)
train_df.loc[train_df["Sex"] == "male", "Sex"] = 0

train_df.loc[train_df["Sex"] == "female", "Sex"] = 1



train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0

train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1

train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2



test_df.loc[test_df["Sex"] == "male", "Sex"] = 0

test_df.loc[test_df["Sex"] == "female", "Sex"] = 1



test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0

test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1

test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2
test_df.sample(10)
train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"] + 1

test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"] + 1
train_df["IsAlone"] = train_df.FamSize.apply(lambda x: 1 if x == 1 else 0)

test_df["IsAlone"] = test_df.FamSize.apply(lambda x: 1 if x == 1 else 0)
for name in train_df["Name"]:

    train_df["Title"] = train_df["Name"].str.extract("([A-Za-z]+)\.",expand=True)

    

for name in test_df["Name"]:

    test_df["Title"] = test_df["Name"].str.extract("([A-Za-z]+)\.",expand=True)

    

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",

          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other", "Rev": "Other", "Dr": "Other"}



train_df.replace({"Title": title_replacements}, inplace=True)

test_df.replace({"Title": title_replacements}, inplace=True)



train_df.loc[train_df["Title"] == "Miss", "Title"] = 0

train_df.loc[train_df["Title"] == "Mr", "Title"] = 1

train_df.loc[train_df["Title"] == "Mrs", "Title"] = 2

train_df.loc[train_df["Title"] == "Master", "Title"] = 3

train_df.loc[train_df["Title"] == "Other", "Title"] = 4



test_df.loc[test_df["Title"] == "Miss", "Title"] = 0

test_df.loc[test_df["Title"] == "Mr", "Title"] = 1

test_df.loc[test_df["Title"] == "Mrs", "Title"] = 2

test_df.loc[test_df["Title"] == "Master", "Title"] = 3

test_df.loc[test_df["Title"] == "Other", "Title"] = 4
train_df.sample(10)
from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV
features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize", "IsAlone", "Title"]

X_train = train_df[features] #define training features set

y_train = train_df["Survived"] #define training label set

X_test = test_df[features] #define testing features set

#we don't have y_test, that is what we're trying to predict with our model
from sklearn.model_selection import train_test_split #to create validation data set



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
svc_clf = SVC() 

svc_clf.fit(X_training, y_training)

pred_svc = svc_clf.predict(X_valid)

acc_svc = accuracy_score(y_valid, pred_svc)



print(acc_svc)
linsvc_clf = LinearSVC()

linsvc_clf.fit(X_training, y_training)

pred_linsvc = linsvc_clf.predict(X_valid)

acc_linsvc = accuracy_score(y_valid, pred_linsvc)



print(acc_linsvc)
rf_clf = RandomForestClassifier()

rf_clf.fit(X_training, y_training)

pred_rf = rf_clf.predict(X_valid)

acc_rf = accuracy_score(y_valid, pred_rf)



print(acc_rf)
logreg_clf = LogisticRegression()

logreg_clf.fit(X_training, y_training)

pred_logreg = logreg_clf.predict(X_valid)

acc_logreg = accuracy_score(y_valid, pred_logreg)



print(acc_logreg)
knn_clf = KNeighborsClassifier()

knn_clf.fit(X_training, y_training)

pred_knn = knn_clf.predict(X_valid)

acc_knn = accuracy_score(y_valid, pred_knn)



print(acc_knn)
gnb_clf = GaussianNB()

gnb_clf.fit(X_training, y_training)

pred_gnb = gnb_clf.predict(X_valid)

acc_gnb = accuracy_score(y_valid, pred_gnb)



print(acc_gnb)
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_training, y_training)

pred_dt = dt_clf.predict(X_valid)

acc_dt = accuracy_score(y_valid, pred_dt)



print(acc_dt)
from xgboost import XGBClassifier

xg_clf = XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)

xg_clf.fit(X_training, y_training)

pred_xg = xg_clf.predict(X_valid)

acc_xg = accuracy_score(y_valid, pred_xg)



print(acc_xg)
model_performance = pd.DataFrame({

    "Model": ["SVC", "Linear SVC", "Random Forest", 

              "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bayes",  

              "Decision Tree", "XGBClassifier"],

    "Accuracy": [acc_svc, acc_linsvc, acc_rf, 

              acc_logreg, acc_knn, acc_gnb, acc_dt, acc_xg]

})



model_performance.sort_values(by="Accuracy", ascending=False)
rf_clf = RandomForestClassifier()



parameters = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], 

              "criterion": ["gini", "entropy"],

              "max_features": ["auto", "sqrt", "log2"], 

              "max_depth": [2, 3, 5, 10], 

              "min_samples_split": [2, 3, 5, 10],

              "min_samples_leaf": [1, 5, 8, 10]

             }



grid_cv = GridSearchCV(rf_clf, parameters, scoring = make_scorer(accuracy_score))

grid_cv = grid_cv.fit(X_train, y_train)



print("Our optimized Random Forest model is:")

grid_cv.best_estimator_
rf_clf = grid_cv.best_estimator_



rf_clf.fit(X_train, y_train)
submission_predictions =rf_clf.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": submission_predictions

    })



submission.to_csv("titanic.csv", index=False)

print(submission.shape)