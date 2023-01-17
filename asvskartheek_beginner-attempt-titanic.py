# General Libraries

import os

print(os.listdir("../input"))



# LinAlg and Dataframes

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline



# Pre-processing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# Models

from sklearn import tree

from sklearn.naive_bayes import GaussianNB



# Scores

from sklearn.metrics import accuracy_score
titanic_dataset = pd.read_csv("../input/train.csv")

titanic_dataset.head()
titanic_dataset.info()
titanic_dataset["Name"].value_counts()
titanic_dataset["Sex"].value_counts()
titanic_dataset["Ticket"].value_counts()
titanic_dataset["Cabin"].value_counts()
titanic_dataset["Embarked"].value_counts()
titanic_dataset.hist(bins=100,figsize=(20,15))

plt.show()
corr_matrix = titanic_dataset.corr()

corr_matrix["Survived"].sort_values(ascending=False)
"""

Dropping the attributes that seems to have absolutely no correlation with Survival of Passenger

"""

titanic_dataset = titanic_dataset.drop("Name",axis=1)

titanic_dataset = titanic_dataset.drop("Cabin",axis=1)

titanic_dataset = titanic_dataset.drop("Ticket",axis=1)

titanic_dataset = titanic_dataset.drop("PassengerId",axis=1)

titanic_dataset.head()
mean_age = titanic_dataset["Age"].mean()

titanic_dataset["Age"].fillna(mean_age,inplace=True)
titanic_dataset.info()
titanic_dataset = titanic_dataset.dropna(subset=["Embarked"])

titanic_dataset.info()
titanic_categorical = titanic_dataset[["Sex","Embarked"]]

titanic_categorical.head(10)
titanic_categorical.info()
titanic_dataset['Sex_Encoded'] = LabelEncoder().fit_transform(titanic_dataset['Sex'])

titanic_dataset[['Sex', 'Sex_Encoded']]
titanic_dataset['Embarked_Encoded'] = LabelEncoder().fit_transform(titanic_dataset['Embarked'])

titanic_dataset[['Embarked', 'Embarked_Encoded']]
titanic_dataset.info()
titanic_dataset = titanic_dataset.drop("Sex",axis=1)

titanic_dataset = titanic_dataset.drop("Embarked",axis=1)

titanic_dataset.info()
X_attr = ["Pclass","Age","SibSp","Parch","Fare","Sex_Encoded","Embarked_Encoded"]

X = titanic_dataset[X_attr]

y_attr = ["Survived"]

y = titanic_dataset[y_attr]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
DT_clf = tree.DecisionTreeClassifier()

DT_clf.fit(X_train,y_train)

DT_preds = DT_clf.predict(test_set)
x = [1,2,3,4,5,6,7,8,9,10]

scores = []

for max_depth in range(1,11):

    DT_clf_tuned = tree.DecisionTreeClassifier(max_depth=max_depth)

    DT_clf_tuned.fit(X_train,y_train)

    DT_clf_tuned_preds = DT_clf_tuned.predict(X_valid)

    scores.append(accuracy_score(y_valid,DT_clf_tuned_preds))

plt.plot(x,scores)

plt.show()

print(scores)
test_set = pd.read_csv("../input/test.csv")

"""

Similar pre-processing with test data as well

1. Drop useless Features

2. Convert Object/Strings to Numerical Type

3. Fill NAs for Numerical Attributes

"""

# Step 1

test_set = test_set.drop("Name",axis=1)

test_set = test_set.drop("Cabin",axis=1)

test_set = test_set.drop("Ticket",axis=1)

test_set = test_set.drop("PassengerId",axis=1)

# Step 2

test_set['Sex_Encoded'] = test_set['Sex'].map( {'male':1, 'female':0} )

test_set['Embarked_Encoded'] = test_set['Embarked'].map( {'C':0, 'Q':1, 'S':2})

test_set = test_set.drop('Sex',axis=1)

test_set = test_set.drop('Embarked',axis=1)

# Step 3

mean_test_age = test_set["Age"].mean()

mean_fare = test_set["Fare"].mean()

test_set["Age"].fillna(mean_test_age,inplace=True)

test_set["Fare"].fillna(mean_fare,inplace=True)
DT_final = tree.DecisionTreeClassifier(max_depth=3)

DT_final.fit(X_train,y_train)

final_preds = DT_final.predict(test_set)
x = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

scores = []

for min_samples_split in range(2,21):

    DT_clf_tuned = tree.DecisionTreeClassifier(max_depth=3,min_samples_split=min_samples_split)

    DT_clf_tuned.fit(X_train,y_train)

    DT_clf_tuned_preds = DT_clf_tuned.predict(X_valid)

    scores.append(accuracy_score(y_valid,DT_clf_tuned_preds))

plt.plot(x,scores)

plt.show()

print(scores)
GNB_clf = GaussianNB()

GNB_clf.fit(X_train,y_train)

GNB_preds = GNB_clf.predict(test_set)