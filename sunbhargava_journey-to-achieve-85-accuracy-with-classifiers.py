#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv('../input/train.csv')

test_df     = pd.read_csv('../input/test.csv')
#Dataset information 

titanic_df.info()

print("----------------------------")

test_df.info()
#checking number of survivors by sex for train datset

sns.set_style('dark')

sns.countplot(data=titanic_df,x='Survived',hue='Sex')
#checking number of survivors by class for train datset

sns.set_style('dark')

sns.countplot(data=titanic_df,x='Survived',hue='Pclass')
#checking if the port of embarkation had anything to do with a person surviving for train dataset

sns.set_style('dark')

sns.countplot(data=titanic_df,x='Survived',hue='Embarked')
#for finding null

sns.heatmap(titanic_df.isnull())

print('Train columns with null values:\n', titanic_df.isnull().sum())

print("-"*10)
sns.heatmap(test_df.isnull())

print('Test columns with null values:\n',test_df.isnull().sum())
#drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_df = titanic_df.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

test_df = test_df.drop(['Name','Cabin','Ticket'],axis=1)
titanic_df.head()

test_df.head()

#Filling missing values in the train dataset

top = titanic_df['Embarked'].describe().top

titanic_df['Embarked'] = titanic_df['Embarked'].fillna(top)
titanic_df.loc[titanic_df.Age.isnull(),'Age'] = titanic_df[~titanic_df.Age.isnull()].Age.mean()

test_df.loc[test_df.Age.isnull(),'Age'] = test_df[~test_df.Age.isnull()].Age.mean()
test_df.loc[test_df.Fare.isnull(),'Fare']=test_df[~test_df.Fare.isnull()].Fare.mean()

#handling categorical variables

titanic_df['Sex'] = titanic_df['Sex'].map({'male' :0, 'female':1})

titanic_df.head

titanic_df['Embarked'] = titanic_df['Embarked'].map({'S' :0,'C':1,'Q':2})

titanic_df.head()
test_df['Sex'] = test_df['Sex'].map({'male' :0, 'female':1})

test_df['Embarked'] = test_df['Embarked'].map({'S' :0,'C':1,'Q':2})

test_df.head()
#Splitting data set for model selection

from sklearn.model_selection import train_test_split

X=titanic_df.drop(['Survived'],axis=1)

y=titanic_df['Survived']

X_test  = test_df.drop("PassengerId",axis=1).copy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train =sc.fit_transform(X_train)

X_test =sc.fit_transform(X_test)
#Using Random Forest for prediction

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

NB =GaussianNB()

NB.fit(X_train,y_train)

y_pred = NB.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

NBcm=confusion_matrix(y_test,y_pred)
#Using SVM for train dataset

from sklearn.svm import SVC

svm_classifier = SVC(kernel='rbf',random_state=0)

svm_classifier.fit(X_train,y_train)

y_pred= svm_classifier.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

SVCcm=confusion_matrix(y_test,y_pred)
# Using Decision tree for train dataset

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion='entropy',random_state=0)

DT.fit(X_train,y_train)

y_pred= DT.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

DTcm=confusion_matrix(y_test,y_pred)
# Using K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train,y_train)

y_pred = knn_classifier.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

KNNcm=confusion_matrix(y_test,y_pred)
# Using LogisticRegression to the Training set

from sklearn.linear_model import LogisticRegression

logistic_clasifier = LogisticRegression(random_state=0)

logistic_clasifier.fit(X_train,y_train)

y_pred= logistic_clasifier.predict(X_test)

#to generate the classification_report

from sklearn.metrics import classification_report , accuracy_score

print(classification_report(y_test, y_pred))

accuracy_score=accuracy_score(y_test, y_pred)

print(accuracy_score)

#confution Matrix to identifiy odd's

from sklearn.metrics import confusion_matrix

LogRegcm=confusion_matrix(y_test,y_pred)