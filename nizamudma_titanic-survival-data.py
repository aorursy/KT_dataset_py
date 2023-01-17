import numpy as np 

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
train_df.info()
test_df.info()
#Dropping some columns which I am assuming is useless for prediction



train_df=train_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test_df=test_df.drop(['Name','Ticket','Cabin'],axis=1)
#replacing SEX to int mapping



train_df['Sex']=train_df['Sex'].map({'male':1,'female':0}).astype(int)

test_df['Sex']=test_df['Sex'].map({'male':1,'female':0}).astype(int)
#filling Embarked and mapping 

print(train_df.describe(include=['O']))



train_df['Embarked']=train_df['Embarked'].fillna("S")



print('-'*20)

print(train_df.describe(include=['O']))



train_df['Embarked']=train_df['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)

test_df['Embarked']=test_df['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)
#filling Age with median age



median_age=train_df['Age'].dropna().median()

train_df['Age']=train_df['Age'].fillna(median_age)

test_df['Age']=test_df['Age'].fillna(median_age)
#filling Fare with median fare : only in test_df



median_fare=test_df['Fare'].dropna().median()

test_df['Fare']=test_df['Fare'].fillna(median_fare)
train_df.columns
test_df.columns
X_train=train_df[[ 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]

Y_train=train_df['Survived']



X_test=test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]



#index

idx=test_df['PassengerId']



X_train.shape , Y_train.shape , X_test.shape , idx.shape
logreg=LogisticRegression()



logreg.fit(X_train,Y_train)

score_logreg=logreg.score(X_train,Y_train)



print("Training Score of Logistic Regression:",score_logreg)



predict_logreg=logreg.predict(X_test)
rfc = RandomForestClassifier(n_estimators=100)



rfc.fit(X_train, Y_train)

score_rfc = rfc.score(X_train, Y_train)



print("Training Score of  RandomForestClassifier:",score_rfc)



out_rfc = rfc.predict(X_test)
knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train, Y_train)

score_knn = knn.score(X_train, Y_train)



print("Training Score of  KNeighborsClassifier:",score_knn)



out_knn = knn.predict(X_test)
svc = SVC()



svc.fit(X_train, Y_train)

score_svc = svc.score(X_train, Y_train)



print("Training Score of  SVM:",score_svc)





out_svc = svc.predict(X_test)  