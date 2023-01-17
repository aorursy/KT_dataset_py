# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing necessary package



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image, display

%matplotlib inline
#load the train dataset

train = pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv')
# display first few of rows

display(train.head())
# import and display test dataset

test = pd.read_csv('../input/machine-learning-on-titanic-data-set/test.csv')

display(test.head())
# examing shape 

train.shape

# check data type,null values

print(train.info())
print(test.info())
# from above data we get to know 'Age','Cabin','Embarked' Having Null values

# also some categorical data we have check occurance of that data

train.describe(include=['O'])
# Removing Feature(not be a correlation between these feature and survival.):

# 1.Name

# 2.Ticket

# 3.cabin (having too many null values)

# 4.PassengerID

# Age is is definitely correlated to survival

# Embarked feature as it may also correlate with survival or another important feature.

# have to combine sibsb and parch to get total count of family members on board

#Analyzing the 'Pclass'features(i.e categorical)

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 'Pclass1 having more corelation with survived rate'

# analyzing 'Sex' features

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# female having higher chance to survived

# these 2 are our assumptions

# Analyzed Continuous features(e.g Age) Using histogram for it

age = sns.FacetGrid(train, col='Survived')

age.map(plt.hist, 'Age', bins=20)
# Survival(0 = No; 1 = Yes)

# children having age less than 5 having high survival rate

# 15-30 age group didn't survived

# oldest passenger near to 80 get survived

# plotting histogram with ordinal variable (p_class)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Pclass', alpha=.5, bins=20)

grid.add_legend();

# now plotting p_class with Age

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# passenger in Pclass 1 2 get survived

# analyzing embarked feature:

grid = sns.FacetGrid(train, col='Survived', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Embarked', alpha=.5, bins=20)

grid.add_legend();
# we are assuming female are having high chances to get survived

# but here embarked c had exception

# analysing fare fetures

train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Higher fare paying passengers had better survival

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# we analyse feature now we dropping feture which is not significant

train_m = train.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1)

test_m = test.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1)

# print new shape

train_m.shape, test_m.shape
# now converting categorical feature 

# consider sex value

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train_m["Sex"] = lb_make.fit_transform(train_m["Sex"])

test_m["Sex"] = lb_make.fit_transform(test_m["Sex"])
# we convert categorical data into numerical by using label encoder

# show column name

print(train_m.columns.values)

print(test_m.columns.values)


# convert Age feature and fill null value with mean



train_m['Age'].fillna((train_m['Age'].mean()), inplace=True)

test_m['Age'].fillna((test_m['Age'].mean()), inplace=True)

# Let us create Age bands and determine correlations with Survived.



train_m['AgeBand'] = pd.cut(train_m['Age'], 5)

test_m['AgeBand'] = pd.cut(test_m['Age'], 5)

train_m[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Let us replace Age with ordinals based on these bands.

combine = [train_m,test_m]

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_m.head()

test_m.head()
#converting age into int 

train_m['Age'] = train_m['Age'].astype(np.int64)

train_m.head()
# same for test

#converting age into int 

test_m['Age'] = test_m['Age'].astype(np.int64)

test_m.head()


# creating new feature by using existing features

# adding sibsp and parch to get exact family size

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_m[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# creating another categorical feature

# using familyzie

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_m[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# now dropping old feaatures from data

train_m = train_m.drop(['Parch', 'SibSp', 'FamilySize','AgeBand'], axis=1)

train_m.head()
test_m = test_m.drop(['Parch', 'SibSp', 'FamilySize','AgeBand'], axis=1)

test_m.head()
# now in embarked having two null values

# filling it with most common 

freq_port = train_m.Embarked.dropna().mode()[0]

print(freq_port)

combine = [train_m]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_m[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(test_m.info())
# converting into categorical feture into numeric

combine = [train_m, test_m]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_m.head()
test_m.head()
# for fare fill missing value

test_m['Fare'].fillna(test_m['Fare'].dropna().median(), inplace=True)

test_m.head()
# checking again for null value

train_m.info()
# checking again for null value

test_m.info()
# fare is  having float data type

# we have to convert it into categorical

# again we divide into patches

train_m['FareBand'] = pd.qcut(train_m['Fare'], 4)



train_m[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
combine = [train_m , test_m]

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_m = train_m.drop(['FareBand'], axis=1)

    

train_m.head(10)
    

test_m.head(10)
test_m.info()
# applying model:

# split data into x &y

X_train = train_m.drop("Survived", axis=1)

Y_train = train_m["Survived"]

X_test  = test_m.copy()

X_train.shape, Y_train.shape,X_test.shape
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
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
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
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
# Model evaluation

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest',  

              'Stochastic Gradient Decent',

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, 

              acc_sgd, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)