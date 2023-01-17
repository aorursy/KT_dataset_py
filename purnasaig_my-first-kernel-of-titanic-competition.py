# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for visualizations

import seaborn as sns #for interactive visualizations

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
#importing datasets

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")

totaldata = [train, test]
#check both train and test datasets

train.head()

test.head()
train.tail()
test.tail()
#finding columns in train data set

train.columns
#the above code gives you in Index form

#this gives you in an array form

train.columns.values
train.columns[0]
train.columns.values[0]
#finding the shape(no.of.rows, no.of.columns) in train

train.shape
#finding the shape(no.of.rows, no.of. columns) in test

test.shape
train.info()
test.info()
train.describe()
#again checking missing values using isna() or isnull() method which most of the people do

train.isna().sum

#using this code snippet it only shows boolean values. if the column contains missing values then 

#it shows True, else it shows False.
train.isna().sum()

#same code as above but using isnull() method

train.isnull().sum()
#from the above code snippet, its clear that we had missing values in Age, Cabin, Embarked columns.

#Cabin has 687 missing values, Age has 177 missing values, Embarked has 2 missing values.

#now try to fill those missing vlues.

test.isnull().sum()
#lets replace missing values of Embarked column in trian data set

#trying to find out what and how many are the values present in Embarked column

train.Embarked.value_counts()
#mostly missing values are replaced with Either of Mean,Median,Mode

#so totally we have 'S' repeated more times which is Mode case.

#so fill the missing 2 values with most repeated value i.e "S"

train["Embarked"]  = train["Embarked"].fillna("S")
#after filling missing values in Embarked column, verifyinig still we have missing values in 

#Embarked column or not.

train.isna().sum()
#so now we do not have any missing values in Embarked columns. 

#fill missing value of Age with Median.in both test and train.

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())
train.info()
test.info()
#lets fill the 1 missing value in Fare column of test data set

test['Fare'] = test["Fare"].fillna("0")
test.info()
#As Cabin Column has most data missing ,

#there wont be any use filling missing values as it would be hard to fill those missing values using

#using any of the techniques like Mean, Medain and  Mode.we have to delete it. 

train = train.drop("Cabin",axis = 1)

test = test.drop("Cabin", axis = 1)

train.info()
test.info()
#now we had another problem, i.e after filling all the missing values, now we have to convert text 

#data to numerical data, because that is what a machine understands at the end of the day.

#for that we have to use label encoder

#but before that lets find can we convert all the text values to numerical ones.

train.Ticket.value_counts()
train['Ticket'].describe()
#its clear that we had 681 unique values, so its impossible to encode it. so delete Ticket column

train = train.drop(['Ticket'],axis = 1)

train.info()
test['Ticket'].describe()
test = test.drop(['Ticket'],axis = 1)

test.info()
#also there is no use with name, so lets delete it

train = train.drop(["Name"], axis = 1)

test = test.drop(['Name'], axis =1)

train.info()
test.info()
#lets encode remaning 2 text columns to numerical ones

#lets write function for label encoder

from sklearn.preprocessing import LabelEncoder
def encode_features(dataset,featurenames):

    for featurename in featurenames:

        LE = LabelEncoder()

        LE.fit(dataset[featurename])

        dataset[featurename] = LE.transform(dataset[featurename])

    return dataset    



train = encode_features(train, ['Sex','Embarked'])

test = encode_features(test, ['Sex','Embarked'])

train.info()
test.info()
#lets convert Fare from float to int

train['Fare'] = train['Fare'].astype(int)

test["Fare"] = test['Fare'].astype(int)

train.info()
test.info()
test.head()
train = train.drop(['PassengerId'], axis=1)

train.head()
X_train = train.drop(['Survived'], axis = 1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_train.head()
Y_train.head()
X_test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
#Logistic regression

LR = LogisticRegression()

LR.fit(X_train, Y_train)

y_pred_LR = LR.predict(X_test)

LR.score(X_train,Y_train)*100

#Support vector machine 

svm = SVC()

svm.fit(X_train, Y_train)

y_pred_svm = svm.predict(X_test)

svm.score(X_train,Y_train)*100
#RandomForestClassirier

rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)

y_pred_rfc = rfc.predict(X_test)

rfc.score(X_train,Y_train)*100
#KNeighborsClassifier

knc = KNeighborsClassifier()

knc.fit(X_train, Y_train)

y_pred_knc = knc.predict(X_test)

knc.score(X_train,Y_train)*100
#KNeighborsClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)

y_pred_dtc = dtc.predict(X_test)

dtc.score(X_train,Y_train)*100
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_dtc

    })

submission.to_csv('submission.csv', index=False)