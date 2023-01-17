#import standard libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

#describe the data

train.describe(include="all")
#here we can see that total observation is 891 and 12 variable.

train.shape
##Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)

##Categorical Features: Survived, Sex, Embarked, Pclass

##Alphanumeric Features: Ticket, Cabin

train.dtypes
#getting the missing values.Age has maximum missing values,but age is crucial variable

train.isnull().sum()
#Treating missing Values

#Replacing the age value by its mean,Dropiing Cabin column as more than 50% value is missing and Replacing Embarked with its mode vale

train['Age'].fillna((train['Age'].mean()), inplace=True)

train=train.drop(['Cabin'],axis=1)

train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])

train.isnull().sum()
#Plotting bar plot for diffrent variable.



#1)Sex-Female are more survived than male

sns.barplot(x="Sex", y="Survived", data=train)



#Print the percentage male vs female by survived

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#2)Pclass-Female are more survived than male

sns.barplot(x="Pclass",y='Survived',data=train)



print("Percengtage of 1st class survived", train['Survived'][train['Pclass']==1].value_counts(normalize = True)[1]*100)

print("Percengtage of 2nd class survived", train['Survived'][train['Pclass']==2].value_counts(normalize = True)[1]*100)

print("Percengtage of 3rd class survived", train['Survived'][train['Pclass']==3].value_counts(normalize = True)[1]*100)          
#creating new dataframe who is surviced

train1=train[train['Survived']==1]
#3)SibSp-Female are more survived than male

sns.barplot(x='SibSp',y='Survived',data=train)



#calculating percentage

print("percentage of each Sibsp who survived"+"\n",train1['SibSp'].value_counts()/train1['SibSp'].count()*100)
#draw a bar plot for Parch vs. survival-People with less than four parents or children aboard are more likely to survive than those with four or more. Again, people traveling alone are less likely to survive than those with 1-3 parents or children.

sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
##Treating missing Values

#Replacing the age value by its mean,Dropiing Cabin column as more than 50% value is missing and Replacing Fare with its mode vale

test['Age'].fillna(train['Age'].mean(),inplace=True)

test=test.drop(['Cabin'],axis=1)

test['Fare']=test['Fare'].fillna(test['Fare'].mode()[0])

#Bucketization of age column

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup']=pd.cut(test['Age'],bins,labels=labels)
#Barplot the Agegroup



sns.barplot(x='AgeGroup',y='Survived',data=train)

plt.show



#percentage of each group

train1=train[train['Survived']==1]

print("percentage of Agegroup who survived""\n", train1["AgeGroup"].value_counts()/train1['AgeGroup'].count()*100)



#we can also drop the Ticket,Name and Fare feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)





#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
#extracting the title from train-name and bucketizing it



train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'] = train['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train['Title'] = train['Title'].replace(['Mlle', 'Miss','Ms'],'Miss')

train['Title'] = train['Title'].replace(['Mme', 'Mrs','Mr'],'Mr')
#extracting the title from test-name and bucketizing it

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test['Title'] = test['Title'].replace(['Mlle', 'Miss','Ms'],'Miss')

test['Title'] = test['Title'].replace(['Mme', 'Mrs','Mr'],'Mr')
#map each Sex value to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Master": 3, "Royal": 4, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)

test['Title'] = test['Title'].map(title_mapping)
#drop Fare values

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)

#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

#map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)