# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Read data from CSV input

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_df, test_df]

print(train_df.columns.values)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df.head()
train_df.info()

print('_'*40)

test_df.info()

#Missing value analysis

#This will help identify how many missing values are in each column and take some suitable corrective action

train_df.isnull().sum()
train_df.describe()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Visualize the data

train_df.hist("Survived", by="Pclass", grid="False", layout=[1,3],figsize = [10,3])

train_df.hist("Survived", by="Sex",figsize = [10,3])

train_df.hist("Survived", by="Embarked", layout=[1,3],figsize = [10,3])

train_df.hist("Age", bins=10, by = ["Survived", "Sex"], layout=[1,4],figsize = [20,3])
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df.head()

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]

train_df.head()

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)





print(train_df["Title"].unique())    

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()

train_df.tail()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


#getting avg age for each tile and assigning avg of that age to same title with missing value



titles = train_df["Title"].unique()

for title in titles:

    avg_age = train_df[((train_df["Title"]==title) & (train_df["Age"].isnull()==False))]["Age"].mean()

    train_df.loc[((train_df["Title"]==title) & (train_df["Age"].isnull()==True)).tolist(),'Age']=round(avg_age)#replace the missing age values in each group with the corresponding average values

    test_df.loc[((test_df["Title"]==title) & (test_df["Age"].isnull()==True)).tolist(),'Age']=round(avg_age)#replace the missing age values in each group with the corresponding average values

    print("Average age for title", title,"is", str(round(avg_age)))#printing average values for each title



train_df.head()
 

train_df["Age"].describe()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
"""

We can now complete the Fare feature for single missing value in test dataset using mode to

get the value that occurs most frequently for this feature. We do this in a single line of code.



Note that we are not creating an intermediate new feature or doing any further analysis for correlation 

to guess missing feature as we are replacing

only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.



We may also want round off the fare to two decimals as it represents currency.

"""



test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head(20)
#Creating fareband feature and analyzing effect of fare on survivability

train_df['FareBand'] = pd.qcut(train_df['Fare'],4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# k-NN



knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
#adaboost with decision stump

adaBoost = AdaBoostClassifier(base_estimator=None,

                              learning_rate=1.0,

                              n_estimators=100)



adaBoost.fit(X_train, Y_train)



Y_pred = adaBoost.predict(X_test)



acc_adaBoost = round(adaBoost.score(X_train, Y_train) * 100, 2)

acc_adaBoost
random_forest = RandomForestClassifier(n_estimators = 80, max_features='auto', criterion='entropy',max_depth=4)

adaboost_random_forest = AdaBoostClassifier(base_estimator=random_forest,

                              learning_rate=1.0,

                              n_estimators=10)



adaboost_random_forest.fit(X_train, Y_train)



Y_pred = adaBoost.predict(X_test)



acc_adaboost_random_forest = round(adaBoost.score(X_train, Y_train) * 100, 2)

acc_adaboost_random_forest
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('/kaggle/working/submission.csv', index=False)