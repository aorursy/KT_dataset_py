# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:00:30 2018

@author: bhgajula
"""

#importing the libraries 
import pandas as pd

#importing the dataset
training_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
survival = training_dataset.iloc[:,1].values

# defining a function to find out to which deck the passenger belongs to from cabin 
def Deck(x): 
    if str(x) != 'nan':
        return str(x)[0] 
    else:
        return 
  
###
#dropping the Survived from training set and appending the test set and removing the passengerid and ticket to make  finest dataset
dataset = training_dataset.drop(columns=["Survived"]).append(test_dataset).drop(columns=["PassengerId", "Ticket"])

#cleaning the data of our dataset
dataset["hasParents"] = dataset["Parch"].apply(lambda x: (x>0)*1)  #making the Parch Column to having Parents and Children or not
dataset["hasSiblings"] = dataset["SibSp"].apply(lambda x: (x>0)*1)  #making the Siblings Column to having Siblings and Spouse or not
dataset["Deck"] = dataset["Cabin"].apply(Deck) #extracting the deck of the passenger by cabin
dataset["Title"] = dataset["Name"].str.extract( ' ([A-Za-z]+\.)', expand= False) #extrcting the title of the passenger
dataset = dataset.drop(columns=["Parch","SibSp","Cabin","Name"], axis=1) #dropping the duplicates columns that are extracted

#Encoding the categorical variable Sex to a binary variable by LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset["Sex"] = le.fit_transform(dataset["Sex"])

#Fillimg out the unavailable port of embarkation as Southhampton(S)
dataset["Embarked"].fillna('S' , inplace = True)

#filling the unknown age values with the median values of age
dataset["Age"].fillna(dataset["Age"].median(), inplace = True )

#filling the unknown fare values with the median values of fare
dataset["Fare"].fillna(dataset["Fare"].median(), inplace = True )

#Dividing the age in to various sets for different age ranges
dataset.loc[(dataset["Age"]<=18), "Age"] = 0
dataset.loc[(dataset["Age"]>18) & (dataset["Age"]<=30),"Age"] = 1
dataset.loc[(dataset["Age"]>30) & (dataset["Age"]<=50),"Age"] = 2
dataset.loc[(dataset["Age"]>50) & (dataset["Age"]<=65),"Age"] = 3
dataset.loc[(dataset["Age"]>65), "Age"]

#Dividing the fare column in to various sets for fare ranges
dataset.loc[(dataset["Fare"]<=7.91), "Fare"]=0
dataset.loc[(dataset["Fare"]>7.91) & (dataset["Fare"]<=14.454), "Fare"]=1
dataset.loc[(dataset["Fare"]>14.454) & (dataset["Fare"]<=31), "Fare"]=2
dataset.loc[(dataset["Fare"])>31, "Fare"]=3

#Converting the fare as int datatype
dataset["Fare"] = dataset["Fare"].astype(int)

#Converting the Pclass as string
dataset["Pclass"] = dataset["Pclass"].astype("str")
    
#OneHotEncoding the dataset
dataset = pd.get_dummies(dataset)

#Taking only 891 records of training set to build a model 
ml_model=training_dataset.shape[0]
X = dataset[:ml_model]
y = survival

#Dividing the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test ,y_train ,y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)


#Building the DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier
classifier0 = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state = 0)
classifier0.fit(X_train,y_train)
#predicting the output
y_pred0 = classifier0.predict(X_test)
#knowing the confusion matrix
from sklearn.metrics import confusion_matrix
cm0=confusion_matrix(y_test,y_pred0)
#accuracy of decision tree classifier
acc_dtc = round(classifier0.score(X_train, y_train) * 100, 2)
acc_dtc
##91.39


###
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(solver='liblinear', random_state = 0)
classifier1.fit(X_train, y_train)
#predicting the output
y_pred1 = classifier1.predict(X_test)
#knowing the confusion matrix
cm1=confusion_matrix(y_test,y_pred1)
#Accuracy of logistic regression
acc_log = round(classifier1.score(X_train, y_train) * 100, 2)
acc_log
##82.65


###
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train)
#predicting the output
y_pred2 = classifier2.predict(X_test)
#knowing the confusion matrix
cm2=confusion_matrix(y_test,y_pred0)
#accuracy of k nearest neighbour
acc_knn = round(classifier2.score(X_train, y_train) * 100, 2)
acc_knn
##86.27


###
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear',gamma='auto', random_state = 0)
classifier3.fit(X_train, y_train)
#predicting the output
y_pred3 = classifier3.predict(X_test)
#knowing the confusion matrix
cm3=confusion_matrix(y_test,y_pred3)
#accuracy of support vector machine classifier
acc_svc = round(classifier3.score(X_train, y_train) * 100, 2)
acc_svc
##79.53


###
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', gamma='auto', random_state = 0)
classifier4.fit(X_train, y_train)
#predicting the output
y_pred4 = classifier4.predict(X_test)
#knowing the confusion matrix
cm4=confusion_matrix(y_test,y_pred4)
#acuracy of kernel svm
acc_ksvm = round(classifier4.score(X_train, y_train) * 100, 2)
acc_ksvm
##79.28


###
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)
#predicting the output
y_pred5 = classifier5.predict(X_test)
#knowing the confusion matrix
cm5=confusion_matrix(y_test,y_pred5)
#accuracy of naive bayes
acc_nb = round(classifier5.score(X_train, y_train) * 100, 2)
acc_nb

###
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier6 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)
#predicting the output
y_pred6 = classifier6.predict(X_test)
#knowing the confusion matrix
cm6=confusion_matrix(y_test,y_pred6)
#accuracy of random forest classifier
acc_rfc = round(classifier6.score(X_train, y_train) * 100, 2)
acc_rfc
##91.39


###
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier7 = XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)
classifier7.fit(X_train, y_train)
#predicting the output
y_pred7 = classifier7.predict(X_test)
#knowing the confusion matrix
cm7=confusion_matrix(y_test,y_pred7)
#accuracy of the xg boost
acc_xgb = round(classifier7.score(X_train, y_train) * 100, 2)
acc_xgb
##85.02


#Now after knowing the best model for predicting the accuracy of survival using it on train.csv
#building the model for getting the results on the train.csv
from xgboost import XGBClassifier
CLASSIFIER = XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)
CLASSIFIER.fit(X,y)

Z = dataset[ml_model:]

Z_pred=CLASSIFIER.predict(Z)

titanic_submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": Z_pred
    })

titanic_submission.to_csv('titanic_submission.csv', index=False)
