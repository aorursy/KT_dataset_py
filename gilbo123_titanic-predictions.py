# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stat

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





#big thanks to Nadin Tamer who got me started on this. 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")





#############################################################################



# create a combined group of both datasets

combine = [train, test]



# extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Sir', 'Dona'], 'Selfish')

    

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Don', 'Major', 

                                                 'Rev', 'Jonkheer', ], 'Hero')#'Dr','Col',  



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Hero": 5, "Selfish": 6, 

                 "Col": 7, "Dr":8}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

#function to put ages in buckets

def simplify_ages(df):

    bins = (0, 3, 10, 18, 23, 30, 40, 50, 60, 120)

    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Dr', 

                   'Col', 'Selfish Adult', 'Adult']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age_cat = categories

    return df



train["Age_cat"] = pd.Series()

test["Age_cat"] = pd.Series()

simplify_ages(train)

simplify_ages(test)





#fill the blanks with average in that category

train = train.set_index("Age_cat")

test = test.set_index("Age_cat")

train['Age'] = train["Age"].fillna(train.groupby(level="Age_cat")['Age'].max())

test['Age'] = test["Age"].fillna(test.groupby(level="Age_cat")['Age'].max())



#print(test[pd.isnull(test["Fare"] == True)])



#add the Fare missing cell (age bracket 5, 3rd class so average 3rd class)

#print(test[test["Pclass"] == 3].mean())

test["Fare"] = test["Fare"].fillna(12.46) #12.46

#print(test[pd.isnull(test["Fare"]) == True])



# train = train.drop(['Fare'], axis = 1)

# test = test.drop(['Fare'], axis = 1)







#fill na's with most common port

train['Embarked'] = train["Embarked"].fillna(3)

test['Embarked'] = test["Embarked"].fillna(3)

#map to numerical value

embark_mapping = {"S": 1, "Q": 2, "C": 3}



train["Embarked"] = train["Embarked"].replace(embark_mapping)

test["Embarked"] = test["Embarked"].replace(embark_mapping)







#same with ticket

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)





#drop the name as this is irrelevant

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)



#convert male and female to 0, 1

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



#new field that is product of age and weighted sex

train["age_sex"] = (train["Sex"] + train["Parch"] - train["SibSp"])*train["Age"]

test["age_sex"] = (test["Sex"] + test["Parch"] - test["SibSp"] )*test["Age"]



#for some reason dropping the age 

#columns help, but not Parch or SibSp???

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)



# train = train.drop(['Parch'], axis = 1)

# test = test.drop(['Parch'], axis = 1)



# train = train.drop(['SibSp'], axis = 1)

# test = test.drop(['SibSp'], axis = 1)





#convert cabin entry to 1, none to 0

train['Cabin'] = train['Cabin'].fillna(0)

test['Cabin'] = test['Cabin'].fillna(0)





#Cabin can be (A, B, C, D, E, F, or G) - assuming A best

cabin_mapping = {r'[A]+': 7, r'[B]+': 6, r'[C]+': 5, r'[D]+': 4, 

                 r'[E]+':3, r'[F]+':2, r'[G]+':0, r'[T]+':0}

train['Cabin'] = train["Cabin"].replace(cabin_mapping, regex=True)

test['Cabin'] = test["Cabin"].replace(cabin_mapping, regex=True)

#convert from string to int64

pd.to_numeric(train['Cabin']).astype('int64')

pd.to_numeric(test['Cabin']).astype('int64')





#######################################################################################



# start the training by setting up data

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)





#Decision tree

decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)





rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_val)

acc_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_rfc)





adaBoost = AdaBoostClassifier()

adaBoost.fit(x_train, y_train)

y_pred = adaBoost.predict(x_val)

acc_adaBoost = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_adaBoost)









#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = rfc.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })



output.describe()



output.to_csv('submission.csv', index=False)