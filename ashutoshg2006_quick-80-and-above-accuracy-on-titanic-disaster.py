# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/train.csv"))
#print(os.listdir("../input/test.csv"))

# Import all the Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

## Some Functions used for the data cleaning and mapping
def family_mapping(n):
    if n<1:
        return 0
    else:
        return 1

# import DataSet
dataset_train=pd.read_csv("../input/train.csv")
dataset_test=pd.read_csv("../input/test.csv")
dataset_test_y=pd.read_csv("../input/gender_submission.csv")
y_test=dataset_test_y['Survived']
## Look at the samples of the dataset
dataset_train.sample(3)
## Lets now Visualize the Embarked vs Survived data
sns.barplot(x='Embarked', y='Survived', hue='Sex', data=dataset_train)
# The Above barplot shows that male or female who embarked in C had a better survival rate
# Also male survived least in Q
## Lets now Visualize the Pclass vs Survived data
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=dataset_train)
# Now lets see the Age plot of the survived passengers
sns.barplot(x='Age', y='Survived', hue="Sex", data=dataset_train)

# To preserve the original train and test dataset, I created a tmp dataset for both train and test, will remove it later
dataset_tmp=dataset_train
## The individual use of Parch and Siblings feature seems unnecessary, both seems to give the same indication that more siblings means better survival chances
# Lets create a Family feature with combining Parch and SibSp
dataset_tmp['Family']=dataset_tmp['Parch']+dataset_tmp['SibSp']
# Now the Family feature is more like eiter travelling alone or with siblings, therefore, lets map the feature with 1 for traveling with Family and 0 for alone
dataset_tmp['Family']=list(map(family_mapping, dataset_tmp['Family'] ))
# Obviously Age is not a great way to visualize the data as its not easy to interpret this data
# However, Age is very important feature, lower and higher age must have better chances of survival
# Therefore, Age is a feature which can give good insight on prediction
## There are few features which have no value in this Model prediction, therefore, lets drop them
# We will drop Ticket, Name of the passanger, and Embarked from this dataset
## The Cabin has mostly nan values and whichever has values dont have more informtaion than the Pclass or Fare. 
# If the dataset is sorted via Survived, it is seen that most of the passanges survived had higher fares and were females, therefore, its a good idea to keep this feature
dataset_tmp = dataset_train.drop(['Parch', 'SibSp','Survived','PassengerId','Ticket', 'Name', 'Embarked', 'Cabin'],1)

# Also, lets map Male and Female of the Sex to 0 and 1
dataset_tmp['Sex'] = dataset_tmp['Sex'].map({'male':1,'female':0})
## There are few nan in Age, therefore, lets fill nan with average age, the reason is that people who survived definitly had age known, however, some of those who didnt still dont have age information
# Therefore, probability is higher that they were more not child or older people, thats why lets replace nan with average Age
dataset_tmp.fillna(dataset_tmp['Age'].mean(), inplace=True)

## Now the training set data is ready
## Now lets do the same thing for test set
dataset_test_tmp=dataset_test
dataset_test.fillna(dataset_test_tmp['Age'].mean(), inplace=True)
dataset_test_tmp['Family']=dataset_test_tmp['Parch']+dataset_test_tmp['SibSp']

dataset_test_tmp = dataset_test_tmp.drop(['Parch', 'SibSp','PassengerId','Ticket', 'Name', 'Embarked', 'Cabin'],1)
dataset_test_tmp['Sex'] = dataset_test_tmp['Sex'].map({'male':1,'female':0})
dataset_test_tmp['Family']=list(map(family_mapping, dataset_test_tmp['Family'] ))
## Now lets create our training variable vectors set x and y
x_train=dataset_tmp
y_train = dataset_train['Survived']
# Same thing for test set, except y_test is created based on gender_submissions.csv
x_test=dataset_test_tmp
## Logistic Regression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
logreg.score(x_train,y_train)
y_pred=logreg.predict(x_test)
## Support Vector Machines
svc=SVC()
svc.fit(x_train, y_train)
svc.score(x_train, y_train)
## K Nearest Neighbors
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn.score(x_train, y_train)
## RandomForestClasifier
rf_Classifier=RandomForestClassifier(n_estimators=300, random_state=0)
rf_Classifier.fit(x_train,y_train)
rf_Classifier.score(x_train,y_train)
nb=GaussianNB()
nb.fit(x_train, y_train)
nb.score(x_train, y_train)
my_submission=pd.DataFrame({'PassengerId': dataset_test_y.PassengerId, 'Survived': y_pred})
my_submission.to_csv('submission.csv', index=False)

