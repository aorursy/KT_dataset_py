# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.Survived[train_data.Sex == "male"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Sex == "female"].value_counts(normalize=True).plot(kind="bar")
# Convert Sex to numerical value

train_data['Sex'] = train_data['Sex'].map({"male": 0, "female": 1}).astype(int)

test_data['Sex'] = test_data['Sex'].map({"male": 0, "female": 1}).astype(int)
# Is there a pattern in the Name Titles?

train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print(train_data['Title'].value_counts())

print(test_data['Title'].value_counts())
train_data.Survived[train_data.Title == "Mr"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Title == "Miss"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Title == "Mrs"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Title == "Master"].value_counts(normalize=True).plot(kind="bar")
# Since Mr, Miss, and Mrs have a pattern give special lables to them and group the rest together

title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }



# Convert Title to numerical value

train_data['Title'] = train_data['Title'].map(title_map).astype(int)

test_data['Title'] = test_data['Title'].map(title_map).astype(int)



# Drop Name

train_data.drop('Name', axis=1, inplace=True)

test_data.drop('Name', axis=1, inplace=True)
train_data.head()
test_data.head()
# Fill in null Ages with median of each Title

train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
# Fill in null AND 0 fare with median of that Pclass (as price goes up with class)

train_data['Fare'] = train_data['Fare'].replace(0,np.NaN)

test_data['Fare'] = test_data['Fare'].replace(0,np.NaN)



train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
# Fill in missing embarked with most popular embark location

# What is the most popular embark location?

print(train_data['Embarked'].value_counts())
# S is the most so use that 

train_data['Embarked'].fillna('S', inplace=True)
# Is there a relationship between embark and survival?

train_data.Survived[train_data.Embarked == "S"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Embarked == "C"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Embarked == "Q"].value_counts(normalize=True).plot(kind="bar")
# Convert Embarked to numerical value

train_data['Embarked'] = train_data['Embarked'].map({"S": 0, "C": 1, "Q": 2}).astype(int)

test_data['Embarked'] = test_data['Embarked'].map({"S": 0, "C": 1, "Q": 2}).astype(int)
train_data
train_data.isnull().sum()
test_data.isnull().sum()
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_data.drop('AgeBand', axis=1, inplace=True)

# Create Age Groups

for data in [train_data, test_data]:

    data.loc[ data['Age'] <= 17, 'Age'] = 0,

    data.loc[(data['Age'] > 17) & (data['Age'] <= 30), 'Age'] = 1

    data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age'] = 2,

    data.loc[(data['Age'] > 50) & (data['Age'] <= 65), 'Age'] = 3,

    data.loc[ data['Age'] > 65, 'Age'] = 4 # Senior 
# Is there a relationship between Age and survival?

for x in [0,1,2,3]:

    plt.show(train_data.Survived[train_data.Age == x].value_counts(normalize=True).plot(kind="bar"))
# Create Fare Groups

# How? look at stats for both combines



# Split fares into 4 parts

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)



# Get median of each part

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

train_data.drop('FareBand', axis=1, inplace=True)

for data in [train_data, test_data]:

    data.loc[ data['Fare'] <= 8, 'Fare'] = 0,

    data.loc[(data['Fare'] > 8) & (data['Fare'] <= 15), 'Fare'] = 1,

    data.loc[(data['Fare'] > 15) & (data['Fare'] <= 30), 'Fare'] = 2,

    data.loc[ data['Fare'] > 30, 'Fare'] = 3
# Is there a relationship between Fare and survival?

for x in [0,1,2,3]:

    plt.show(train_data.Survived[train_data.Fare == x].value_counts(normalize=True).plot(kind="bar"))
# Combine SibSp & Parch into a new column to get the size of a family

train_data["Family"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data["Family"] = test_data["SibSp"] + test_data["Parch"] + 1



# Is there a relationship between family size and survival?

train_data["Family"].value_counts()
train_data.Survived[train_data.Family == 1].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Family == 2].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Family == 3].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Family == 4].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Family == 5].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Family == 6].value_counts(normalize=True).plot(kind="bar")
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

train_data['Family'] = train_data['Family'].map(family_mapping)

test_data['Family'] = test_data['Family'].map(family_mapping)
# drop unused columns

train_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('SibSp', axis=1, inplace=True)



train_data.drop('Parch', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)



train_data.drop('PassengerId', axis=1, inplace=True)



# Drop Ticket as information is not needed

train_data.drop('Ticket', axis=1, inplace=True)

test_data.drop('Ticket', axis=1, inplace=True)



# Drop Cabin as there are too many empty rows

# train_data.drop('Cabin', axis=1, inplace=True)

# test_data.drop('Cabin', axis=1, inplace=True)
train_data.Cabin.value_counts()
# Lets quantify Cabin by the first letter

train_data['Cabin'] = train_data['Cabin'].str[:1]

test_data['Cabin'] = test_data['Cabin'].str[:1]



train_data.Cabin.value_counts()
# Is there a relationship between cabin and survival

train_data.Survived[train_data.Cabin == "C"].value_counts(normalize=True).plot(kind="bar")
train_data.Survived[train_data.Cabin == "B"].value_counts(normalize=True).plot(kind="bar")

train_data.Survived[train_data.Cabin == "D"].value_counts(normalize=True).plot(kind="bar")

train_data.Survived[train_data.Cabin == "E"].value_counts(normalize=True).plot(kind="bar")

train_data.Survived[train_data.Cabin == "A"].value_counts(normalize=True).plot(kind="bar")

train_data.Survived[train_data.Cabin == "F"].value_counts(normalize=True).plot(kind="bar")
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

train_data['Cabin'] = train_data['Cabin'].map(cabin_mapping)

test_data['Cabin'] = test_data['Cabin'].map(cabin_mapping)



# fill in null

train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train_data
test_data
# Model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV





kfold = KFold(10, True, 1)

X = train_data.drop("Survived", axis=1)

y = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()





# kNN

model = KNeighborsClassifier(n_neighbors = 13)

score = cross_val_score(model, X, y, cv=kfold, n_jobs=1, scoring='accuracy')

print("kNN: " + str(round(np.mean(score)*100, 2)))



# Naive Bayes

model = GaussianNB()

score = cross_val_score(model, X, y, cv=kfold, n_jobs=1, scoring='accuracy')

print("Naive Bayes: " + str(round(np.mean(score)*100, 2)))



# Decision Tree

model = DecisionTreeClassifier()

score = cross_val_score(model, X, y, cv=kfold, n_jobs=1, scoring='accuracy')

print("Decision Tree: " + str(round(np.mean(score)*100, 2)))



# Logistic Regression

model = LogisticRegression()

score = cross_val_score(model, X, y, cv=kfold, n_jobs=1, scoring='accuracy')

print("Logistic Regression: " + str(round(np.mean(score)*100, 2)))



# Random Forest

model = RandomForestClassifier(n_estimators=100) #, max_depth=8, random_state=1

score = cross_val_score(model, X, y, cv=kfold, n_jobs=1, scoring='accuracy')

print("Random Forest: " + str(round(np.mean(score)*100, 2)))
# Commented out to run faster

# rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

# param_grid = {"min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10], "max_depth" : [4, 7, 8, 9], "n_estimators": [50, 100, 400]}

# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# gs = gs.fit(train_data.iloc[:, 1:], train_data.iloc[:, 0])



# print(gs.best_score_)

# print(gs.best_params_)
model = RandomForestClassifier(criterion='gini', 

                             n_estimators=50,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_depth=7,

                             random_state=1)

model.fit(X,y)

predictions = model.predict(X_test)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")