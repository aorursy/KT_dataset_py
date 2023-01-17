# Data analysis
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



print(train) #showing train

print(test) #showing test
train.columns #showing columns of train
test.columns #showing columns of test
train["Survived"].value_counts() #how many survived and didn't survive in train dataset
train.info() #information about data in train (type), (number of non null data)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for col in train:  #We try to find missing values in train dataset

    val=train[col].isnull().sum()

    if val>0.0:

        print("Number of missing values in column ",col,":",val)
for col in test: #We try to find missing values in test dataset

    val=test[col].isnull().sum()

    if val>0.0:

        print("Number of missing values in column ",col,":",val)
combine = [train, test]; print(combine) # We combine two datasets
combine = [train, test]  #before droping some features in combine dataset

print("Before","train(Rows,col)=", combine[0].shape,"test(Rows,col)=", combine[1].shape)
train = train.drop(['Ticket','PassengerId'], axis=1) # we drop here ticket and passengerID from train

test = test.drop(['Ticket'], axis=1) # we drop here ticket from test

combine = [train, test]



print("After","train(Rows,col)=", combine[0].shape, "test(Rows,col)=",combine[1].shape) #checking number of columns and rows after droping
port = train.Embarked.dropna().mode()[0] #Finding most frequent port people ported from it in train dataset

port  
port = test.Embarked.dropna().mode()[0] #Finding most frequent port people ported from it in test dataset

port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(port)

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

(combine[0].isnull().sum(),   

combine[1].isnull().sum()) 
combine[1].isnull().any() # rechecking if there's any missing data
for dataset in combine: #We extract tittles from name 

    dataset['Name'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Name'], train['Sex']) #We see the number of females and males with this titles
for dataset in combine:  #we replace it with names

    dataset['Name'] = dataset['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



print(dataset['Name'])
train[['Name', 'Survived']].groupby(['Name'], as_index=False).mean() # Relation between survial rate and names
name_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5} #We factorize names and sex  

for dataset in combine:

    dataset['Name'] = dataset['Name'].map(name_mapping)

    dataset['Name'] = dataset['Name'].fillna(0)

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



print(combine)
for dataset in combine: #We map Emabraked with numerical values

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)#We see the relation here between survivial rate and Embraked 
# Creating new features
for dataset in combine: #We create new feature by combining Parch and SibSp and we see the relation between survivsl rate and Family

    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine: #factorizing family feature

    dataset['Family'] = dataset['Family'].map( {1: 0,2:1,3:1,4:1,5:0,6:0,7:0,8:0,11:0 } ).astype(int)

train.head()
train['Agegroup'] = pd.cut(train['Age'], 5) #We create here new feature by grouping age and seeing the relation between different groups of age and survival rate

train[['Agegroup', 'Survived']].groupby(['Agegroup'], as_index=False).mean().sort_values(by='Agegroup', ascending=True)
for dataset in combine:     #Based on Agegroup Feature we factorize our Age feature in combine dataset

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train.head()

train['Farerange'] = pd.qcut(train['Fare'], 4) #We create here new feature by grouping Fare and seeing the relation between different ranges of Fares and survival rate

train[['Farerange', 'Survived']].groupby(['Farerange'], as_index=False).mean().sort_values(by='Farerange', ascending=True)
for dataset in combine: #Based on Farerange Feature we factorize our Fare feature in combine dataset

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train.head(10)
# Models and prediction
train = train.drop(['SibSp', 'Parch','Farerange','Cabin','Agegroup'], axis=1) 

test = test.drop(['SibSp', 'Parch','Cabin'], axis=1)
X = train.drop("Survived", axis=1) #We define X as train dataset without survived rate

Y= train["Survived"]  # We define Y to be survived of data

Xt = test.drop("PassengerId", axis=1).copy() #We define Xt to be test dataset without PassengerId 

X.shape, Y.shape, Xt.shape #We check we have the same number of columns
# Random Forest Model

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, Y)

Y_pred = random_forest.predict(Xt)

random_forest.score(X, Y)

acc_random_forest = round(random_forest.score(X, Y) * 100, 2)

acc_random_forest
# Support Vector Machines Model



from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X, Y)

Y_pred = svc.predict(Xt)

acc_svc = round(svc.score(X, Y) * 100, 2)

acc_svc
# Logistic Regression Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X, Y)

Y_pred = logreg.predict(Xt)

acc_log = round(logreg.score(X, Y) * 100, 2)

acc_log
#k-Nearest Neighbors model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X, Y)

Y_pred = knn.predict(Xt)

acc_knn = round(knn.score(X, Y) * 100, 2)

acc_knn
#Deision Tree Model

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X, Y)

Y_pred = decision_tree.predict(Xt)

acc_decision_tree = round(decision_tree.score(X, Y) * 100, 2)

acc_decision_tree



submission1 = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred })
submission1
#submission1.to_csv('../output/submission.csv', index=False) #saveing our submission sheet