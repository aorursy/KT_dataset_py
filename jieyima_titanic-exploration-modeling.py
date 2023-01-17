import pandas as pd

import numpy as np

import matplotlib 

import plotly as plt

import seaborn as sns

import sklearn

from functools import reduce
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
## get an overview of the train dataset

train_df.head(5)
## checking null status for each column

train_df.isnull().sum()
print('Oldest Passenger was:',round(train_df['Age'].max(),2),'Years')

print('Youngest Passenger was:',round(train_df['Age'].min(),2),'Years')

print('Average Age of Passenger was:',round(train_df['Age'].mean(),2),'Years')
train_df['Initial']=0

for InitialOfPassenger in train_df:

    train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.') 
train_df.groupby('Initial')['Age'].count()
train_df.groupby('Initial')['Age'].mean()
train_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Mr','Mr','Mr','Mr'],inplace=True)
train_df.groupby('Initial')['Age'].mean()
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=4.574167

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']= 21.86

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=32.89

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=35.98

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=51.33
train_df["Embarked"].fillna("S",inplace=True)
## checking if the dataset has any null value

train_df.isnull().any()
# Mapping Sex

train_df['Sex'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# Mapping Embarked

train_df['Embarked'] = train_df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
# Mapping Age

train_df['Age_band']=0

train_df.loc[ train_df['Age'] <= 16, 'Age_band'] = 0

train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age_band'] = 1

train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age_band'] = 2

train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age_band'] = 3

train_df.loc[ train_df['Age'] > 64, 'Age_band'] = 4
# Mapping SibSp and Parch

train_df["WithFamily"]= 0

train_df.loc[train_df['SibSp'].astype(int) + train_df['Parch'].astype(int)<=0, 'WithFamily'] = 0

train_df.loc[train_df['SibSp'].astype(int) + train_df['Parch'].astype(int)>=1, 'WithFamily'] = 1
# Mapping titles

train_df['Title'] = 0

train_df['Title'] = train_df['Initial'].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Other":5}).astype(int)

train_df.head()
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']

train_df = train_df.drop(drop_elements, axis = 1)

train_df = train_df.drop(['Initial'], axis = 1)

test_df  = test_df.drop(drop_elements, axis = 1)
#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train,test=train_test_split(train_df,test_size=0.2,random_state=0,stratify=train_df['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=train_df[train_df.columns[1:]]

Y=train_df['Survived']
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y.values.ravel())

prediction1=model.predict(test_X)

print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y.values.ravel())

prediction2=model.predict(test_X)

print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y.values.ravel())

prediction3=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction4=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))
model=KNeighborsClassifier() 

model.fit(train_X,train_Y.values.ravel())

prediction5=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_Y.values.ravel())

prediction6=model.predict(test_X)

print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','Decision Tree','KNN','Random Forest']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),

        KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

models_dataframe
# Generate Predictions

randomforest = RandomForestClassifier()

prediction = randomforest.predict(test)



submission_titanic = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': prediction })

#print(submission_titanic)

submission_titanic.to_csv("submission_titanic.csv", index = False)