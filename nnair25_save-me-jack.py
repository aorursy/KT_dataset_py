# This section updated as and when required #



#Data analysis and wrangling

import numpy as np

import pandas as pd



#Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns



#Machine Learning

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR





train = pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')

print(train.head(5))
print(train.columns.values)

print('-'*50)

train.info()

print('-'*50)

test.info()







#For Numerical Features

print ("N Features")

print (train.describe())

print('-'*50)

#For Categorical Features

print ("C Features")

print (train.describe(include=['O']))

#Check unique values in all columns

for column in train:

    uni=train[column].unique()

    print("No of unique values of ", column, ":", len(uni), "\n" )

   
X=train.dropna(axis=0)#Null values create problem in predicting data

y=X['Survived'].copy()

X=X.drop(labels=['PassengerId','Survived'],axis=1)#Dropping PassengerId as well as it is inconsequential

X=pd.get_dummies(X,columns=['Sex',  'Ticket' ,'Cabin', 'Embarked','Name'])

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=7)

#model= tree.DecisionTreeClassifier()

model=RandomForestClassifier(random_state=7)

model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)

print("Accuracy of RandomForest(Will vary according to random state)=",accuracy*100,'%')
PClassSurvived=train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)

print(PClassSurvived)

sns.barplot(x='Pclass', y='Survived', data=PClassSurvived)
AgeSurvived= train[['Age','Survived']].groupby('Age',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(AgeSurvived)

fig, axis=plt.subplots(1,1,figsize=(18,8))

AgeSurvived["Age"] = AgeSurvived["Age"].astype(int)

sns.barplot(x='Age',y='Survived',data=AgeSurvived)
SibSpSurvived= train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(SibSpSurvived)

sns.barplot(x='SibSp',y='Survived',data=SibSpSurvived)
ParchSurvived= train[['Parch','Survived']].groupby('Parch',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(ParchSurvived)

sns.barplot(x='Parch',y='Survived',data=ParchSurvived)
SexSurvived= train[['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(SexSurvived)

sns.barplot(x='Sex',y='Survived',data=SexSurvived)
EmbarkedSurvived= train[['Embarked','Survived']].groupby('Embarked',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(EmbarkedSurvived)

sns.barplot(x='Embarked',y='Survived',data=EmbarkedSurvived)
#Let's combine SibSp and Parch simply as a new column family members

train['family_size']=train['SibSp']+ train['Parch'] + 1

test['family_size']=test['SibSp']+ test['Parch'] + 1

FSurvived= train[['family_size','Survived']].groupby('family_size',as_index=False).mean().sort_values(by='Survived',ascending=False)

print(FSurvived)

sns.barplot(x='family_size',y='Survived',data=FSurvived)
#Extract Title from Names and group them

train['Title']=train['Name'].str.extract('([A-Za-z]+)\.',expand=False)

test['Title']=test['Name'].str.extract('([A-Za-z]+)\.',expand=False)

pd.crosstab(train['Title'],train['Sex'])
#Let's do combining of the titles

train['Title']=train['Title'].replace(['Lady', 'Countess','Capt', 'Col',

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')



test['Title']=test['Title'].replace(['Lady', 'Countess','Capt', 'Col',

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')



TSurvived=train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False)

print(TSurvived)

sns.barplot(x='Title',y='Survived',data=TSurvived)
TicketSurvived=train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Ticket',ascending=False)

print(TicketSurvived)
print(train[['Cabin']].info())

print('-'*50)

print(test[['Cabin']].info())
train=train.drop(['Cabin','Ticket'],axis=1)

test=test.drop(['Cabin','Ticket'],axis=1)

train=train.dropna(axis=0,subset=['Embarked'])

test['Fare']=test['Fare'].fillna(test['Fare'].mean())

test.info()
train['Age']=train['Age'].fillna(train['Age'].mean())

test['Age']=test['Age'].fillna(test['Age'].mean())

train.loc[ train['Age'] <= 16, 'Age'] = 0

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1

train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2

train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3

train.loc[ train['Age'] > 64, 'Age'] = 4   



test.loc[ test['Age'] <= 16, 'Age'] = 0

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

test.loc[ test['Age'] > 64, 'Age'] = 4   



print(train['Age'].unique())
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2

train.loc[ train['Fare'] > 31, 'Fare'] = 3

train['Fare'] = train['Fare'].astype(int)

test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0

test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1

test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2

test.loc[ test['Fare'] > 31, 'Fare'] = 3

test['Fare'] = test['Fare'].astype(int)



print(train['Fare'].unique())
train['Sex']=train['Sex'].map({'female':0,'male':1}).astype(int)

train['Title']=train['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)

train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



test['Sex']=test['Sex'].map({'female':0,'male':1}).astype(int)

test['Title']=test['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)

test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train.dtypes)
y=train['Survived'].copy()

X=train.drop(labels=['PassengerId','Survived','Name'],axis=1)#Dropping PassengerId, Name it is inconsequential

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=7)

#model= tree.DecisionTreeClassifier()

model=RandomForestClassifier(random_state=15)

model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)

print("Accuracy of RandomForest(Will vary according to random state)=",accuracy*100,'%')
XTest=test.drop(labels=['PassengerId','Name'],axis=1)

Y_pred=model.predict(XTest)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)




