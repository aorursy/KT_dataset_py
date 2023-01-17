# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print(test_data.keys())

print(train_data.keys())
#print(train_data.dtypes)

types_train = train_data.dtypes

print(types_train)

num_types = types_train[types_train == float]

#print(num_types)
train_data.describe()
test_data.describe()
print("Train data Frame")

train_data.count()

print(train_data.isnull().sum())
print(test_data.isnull().sum())
train_data.drop(labels = ["Cabin","Ticket"], axis = 1, inplace=True)

test_data.drop(labels = ["Cabin","Ticket"], axis = 1, inplace=True)

print(set(train_data["Embarked"]))

print(set(train_data["Sex"]))
#sns.distplot(test_data["Age"])

copy = train_data.copy()

copy.dropna(inplace=True)

sns.distplot(copy["Age"])
train_data["Age"].fillna(train_data["Age"].median(), inplace = True)

test_data["Age"].fillna(train_data["Age"].median(), inplace = True)

test_data["Fare"].fillna(train_data["Fare"].mean(), inplace = True)

train_data.dropna(inplace=True)

#test_data.dropna(inplace=True)

test_data.count()
train_data.head()
test_data.head()
sns.barplot(x="Sex", y="Survived", data=train_data)

plt.title("Distribution based on gender")

plt.show()

total_survived_females = train_data.loc[train_data["Sex"] == "female"]["Survived"].sum()

total_survived_males = train_data.loc[train_data["Sex"] == "male"]["Survived"].sum()

total_survivals = total_survived_females + total_survived_males



print("Total people survived is: " + str((total_survived_females + total_survived_males)))

print("Proportion of Females who survived:") 

print(total_survived_females/(total_survived_females + total_survived_males))

print("Proportion of Males who survived:")

print(total_survived_males/(total_survived_females + total_survived_males))

women = train_data.loc[train_data.Sex == 'female']["Survived"]

women.head(20)

rate_women = sum(women)/len(women)

print("% of women who survived: ", rate_women)
sns.barplot(x="Pclass", y="Survived", data = train_data)

plt.title("Distribution of Survival based on Class")

plt.show()
total_survived_one = train_data[train_data.Pclass == 1]["Survived"].sum()

total_survived_two = train_data[train_data.Pclass == 2]["Survived"].sum()

total_survived_three = train_data[train_data.Pclass == 3]["Survived"].sum()

total_survived = total_survived_one + total_survived_two + total_survived_three

print(f"Total survivors {total_survived}")

print(f"Proportion of first class passengers amongst survivors: {total_survived_one/total_survived}")

print(f"Proportion of second class passengers amongst survivors: {total_survived_two/total_survived}")

print(f"Proportion of third class passengers amongst survivors: {total_survived_three/total_survived}")
sns.barplot(x="Pclass", y="Survived", hue="Sex", data = train_data)

plt.title("Survival Rate based on Gender and Class")

plt.show()
sns.barplot(x="Sex", y="Survived", hue="Pclass", data = train_data)

plt.title("Survival Rate based on Gender and Class")

plt.show()
survived_ages = train_data[train_data.Survived == 1]["Age"]

not_survived_ages = train_data[train_data.Survived == 0]["Age"]

plt.subplot(1,2,1)

sns.distplot(survived_ages, kde = False)

plt.axis([0, 100, 0, 150])

plt.title("Survived")

plt.subplot(1,2,2)

sns.distplot(not_survived_ages, kde = False)

plt.title("Deceased")

plt.axis([0, 100, 0, 150])

plt.show()
sns.stripplot(x = "Survived", y = "Age", data = train_data, jitter = True)
sns.pairplot(train_data)
train_data.sample(5)
print(set(train_data["Embarked"]))

print(set(train_data["Sex"]))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(train_data["Sex"])

train_data["Sex"] = le.transform(train_data["Sex"])

test_data["Sex"] = le.transform(test_data["Sex"])

le_embarked = LabelEncoder()

le_embarked.fit(train_data["Embarked"])

train_data["Embarked"] = le_embarked.transform(train_data["Embarked"])

test_data["Embarked"] = le_embarked.transform(test_data["Embarked"])

train_data.sample(5)
train_data["FamSize"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data["FamSize"] = test_data["SibSp"] + test_data["Parch"] + 1
train_data["IsAlone"] = train_data.FamSize.apply(lambda x: 1 if x == 1 else 0)

test_data["IsAlone"] = test_data["FamSize"].apply(lambda x: 1 if x == 1 else 0)

test_data.head(5)
train_data["Title"] = train_data["Name"].str.extract("([A-Za-z]+)\.",expand=True)

test_data["Title"] = test_data["Name"].str.extract("([A-Za-z]+)\.",expand=True)

train_data.head(5)
test_data.head(5)
titles = set(train_data["Title"])

print(titles)
from collections import Counter

title_counts = Counter(train_data["Title"])

#for title, count in title_counts.items():

#    print(title, count)

title_df = pd.DataFrame.from_dict(title_counts, orient='index').reset_index()

print(title_df)
title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",

          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}



train_data["Title"].replace(title_replacements, inplace=True)

test_data["Title"].replace(title_replacements, inplace=True)

from collections import Counter

title_counts = Counter(train_data["Title"])

#for title, count in title_counts.items():

#    print(title, count)

title_df = pd.DataFrame.from_dict(title_counts, orient='index').reset_index()

print(title_df)
le_title = LabelEncoder()

le_title.fit(train_data["Title"])

train_data["Title"] = le_title.transform(train_data["Title"])

test_data["Title"] = le_title.transform(test_data["Title"])

train_data.drop("Name",  axis=1 , inplace = True)

test_data.drop("Name",  axis=1 , inplace = True)

train_data.sample(5)
test_data.sample(5)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

ages_train = np.array(train_data["Age"]).reshape(-1,1)

fares_train = np.array(train_data["Fare"]).reshape(-1,1)

ages_test = np.array(test_data["Age"]).reshape(-1,1)

fares_test = np.array(test_data["Fare"]).reshape(-1,1)

train_data["Age"] = scaler.fit_transform(ages_train)

test_data["Age"] = scaler.fit_transform(ages_test)

train_data["Fare"] = scaler.fit_transform(fares_train)

test_data["Fare"] = scaler.fit_transform(fares_test)
train_data.sample(5)

train_data.dtypes
from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
X_train = train_data.drop(labels=["PassengerId","Survived"], axis = 1)

y_train = train_data["Survived"]

X_test = test_data.drop(["PassengerId"], axis = 1)

X_train.sample(5)
from sklearn.model_selection import train_test_split

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state = 0)
svc_clf = SVC()

parameters_svc = {"kernel": ["rbf", "linear"], "probability": [True, False], "verbose": [True, False]}

grid_svc = GridSearchCV(svc_clf, parameters_svc, scoring = make_scorer(accuracy_score))

grid_svc.fit(X_training, y_training)

svc_clf = grid_svc.best_estimator_

svc_clf.fit(X_training, y_training)

pred_svc = svc_clf.predict(X_valid)

acc_svc = accuracy_score(y_valid, pred_svc)

print("The Score for SVC is: " + str(acc_svc))
linsvc_clf = LinearSVC()



parameters_linsvc = {"multi_class": ["ovr", "crammer_singer"], "fit_intercept": [True, False], "max_iter": [100, 500, 1000, 1500]}



grid_linsvc = GridSearchCV(linsvc_clf, parameters_linsvc, scoring=make_scorer(accuracy_score))

grid_linsvc.fit(X_training, y_training)



linsvc_clf = grid_linsvc.best_estimator_



linsvc_clf.fit(X_training, y_training)

pred_linsvc = linsvc_clf.predict(X_valid)

acc_linsvc = accuracy_score(y_valid, pred_linsvc)



print("The Score for LinearSVC is: " + str(acc_linsvc))
rf_clf = RandomForestClassifier()



parameters_rf = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"], 

                 "max_depth": [2, 3, 5, 10], "min_samples_split": [2, 3, 5, 10]}



grid_rf = GridSearchCV(rf_clf, parameters_rf, scoring=make_scorer(accuracy_score))

grid_rf.fit(X_training, y_training)



rf_clf = grid_rf.best_estimator_



rf_clf.fit(X_training, y_training)

pred_rf = rf_clf.predict(X_valid)

acc_rf = accuracy_score(y_valid, pred_rf)



print("The Score for Random Forest is: " + str(acc_rf))
model_performance = pd.DataFrame({"Model":["SVC", "Linear SVC", "Random Forest"],"Accuracy":[acc_svc, acc_linsvc, acc_rf]})

model_performance.sort_values(by="Accuracy",ascending = False)
print(test_data.shape)

svc_clf.fit(X_train, y_train)

submission_predictions = svc_clf.predict(X_test)

submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": submission_predictions})

submission.to_csv("titanic.csv", index = False)

print(submission.shape)