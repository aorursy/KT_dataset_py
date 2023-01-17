#LIBRARIES

import os

import numpy as np

import pandas as pd



#VISUALIZATION

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#MACHINE LEARNING

from sklearn import svm

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
#importing data 

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df,test_df]

train_df.head()
train_df.tail()
train_df.describe(include='all')
#Parch and Survival

train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#SibSp and survival

train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Sex and Survival

train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Pclass and Survival

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Age vs Survival

g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist,'Age',bins=20)
# Age and Pclass vs Survival 

g=sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.5,aspect=2)

g.map(plt.hist,'Age',bins=20)
# adding Sex, Embarked and Pclass to the model

grid=sns.FacetGrid(train_df,row='Embarked',size=2.5,aspect=1.6)

grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')

grid.add_legend()
#visualising based on Sex, Fare

grid=sns.FacetGrid(train_df,row='Embarked',col='Survived',size=2.2,aspect=1.6)

grid.map(sns.barplot, 'Sex','Fare',ci=None)

grid.add_legend()
print('Before', train_df.shape, test_df.shape,combine[0].shape,combine[1].shape)



train_df= train_df.drop(['Ticket','Cabin'],axis=1)

test_df= test_df.drop(['Ticket','Cabin'],axis=1)

combine= [train_df,test_df]



print('After', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset=dataset.drop(['Name'],axis=1)

    

combine=[train_df,test_df]
#converting categorical values to numerical

map1={'male':0,'female':1}

map2={'S':1,'Q':2,'C':3}

train_df=train_df.replace({'Sex':map1,'Embarked':map2})

test_df=test_df.replace({'Sex':map1,'Embarked':map2})

combine=[train_df,test_df]



train_df.describe()
#Filling the missing values in Age

g=sns.FacetGrid(train_df,row='Embarked',col='Survived',size=4,aspect=2)

g.map(sns.pointplot,'Sex','Age',palette='deep')

g.add_legend()
#filling the missing values of Age from the above observation with median

for dataset in combine:

    guess_age=dataset[:]['Age'].median()

train_df[:]['Age']=train_df[:]['Age'].fillna(guess_age)

test_df[:]['Age']=test_df[:]['Age'].fillna(guess_age)

combine=[train_df,test_df]

for dataset in combine:

    dataset.describe()

    

for dataset in combine:

    guess=dataset[:]['Embarked'].median()

train_df[:]['Embarked']=train_df[:]['Embarked'].fillna(guess)

combine=[train_df,test_df]

for dataset in combine:

    dataset.describe()

    

for dataset in combine:

    guess_fare=dataset[:]['Fare'].median()

test_df[:]['Fare']=train_df[:]['Fare'].fillna(guess_fare)

combine=[train_df,test_df]

for dataset in combine:

    dataset.describe()
#creating a new feature for age called Age bands

train_df['AgeBand']=pd.cut(train_df['Age'],5)

train_df[['AgeBand','Survived']].groupby(('AgeBand'),as_index=False).mean().sort_values(by='AgeBand',ascending=True)
#categorizing into age bands

for dataset in combine:

    dataset.loc[dataset['Age']<=16,'Age'] = 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age'] = 1

    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age'] = 2

    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age'] = 3

    dataset.loc[dataset['Age']>64,'Age'] = 4



combine=[train_df,test_df]

train_df.head()

    
#dropping the AgeBand feature

train_df=train_df.drop(['AgeBand','Name'],axis=1)

combine=[train_df,test_df]

train_df.head()
#create one more feature combining SibSp and Parch

for dataset in combine:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1

train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)

combine=[train_df,test_df]

train_df.head()
#from above we can create a feature for whether a person is alone

for dataset in combine:

    dataset.loc[dataset['FamilySize']<=1,'IsAlone'] = 1

    dataset.loc[dataset['FamilySize']>1,'IsAlone'] = 0

    

combine=[train_df,test_df]

train_df.head()
#dropping redundant features namely SibSp, Parch, FamilySize

train_df=train_df.drop(['SibSp','Parch','FamilySize'],axis=1)

combine=[train_df,test_df]

train_df.head()
#similar to the AgeBand we can also create FareBand

train_df['FareBand']=pd.cut(train_df['Fare'],5)

train_df[['FareBand','Survived']].groupby('FareBand',as_index=True).mean().sort_values(by='Survived',ascending=False)
#converting into numerical categories

combine=[train_df,test_df]

for dataset in combine:

    dataset.loc[dataset['Fare']<=102,'Fare'] = 1

    dataset.loc[(dataset['Fare']>102) & (dataset['Fare']<=205), 'Fare'] = 2

    dataset.loc[(dataset['Fare']>205) & (dataset['Fare']<=308), 'Fare'] = 3

    dataset.loc[(dataset['Fare']>308) & (dataset['Fare']<=410), 'Fare'] = 4

    dataset.loc[dataset['Fare']>410, 'Fare'] = 5

combine=[train_df,test_df]



train_df.head()
#dropping FareBand

train_df=train_df.drop(['FareBand'],axis=1)

combine=[train_df,test_df]



train_df.head(10)
test_df.head(10)
test_df=test_df.drop(['Name','SibSp','Parch'],axis=1)

test_df.head()
# creating train and test sets

X_train=train_df.drop('Survived',axis=1)

Y_train=train_df['Survived']

X_test=test_df.drop('PassengerId',axis=1)



X_train.shape, Y_train.shape, X_test.shape
#Logistic Regression

logreg=LogisticRegression()

logreg.fit(X_train,Y_train)

Y_predict=logreg.predict(X_test)

acc_log=round(logreg.score(X_train,Y_train)*100,2)

acc_log
# Support Vector Machine

svc = SVC()

svc.fit(X_train,Y_train)

Y_predict=svc.predict(X_test)

acc_svc=round(svc.score(X_train,Y_train)*100,2)

acc_svc
# KNN

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)

Y_predict=knn.predict(X_test)

acc_knn=round(knn.score(X_train,Y_train)*100,2)

acc_knn
# Gaussian Naive Bayes

gaussian=GaussianNB()

gaussian.fit(X_train,Y_train)

Y_predict=gaussian.predict(X_test)

acc_gaussian=round(gaussian.score(X_train,Y_train)*100,2)

acc_gaussian
# Random Forest Classifier

rfc=RandomForestClassifier()

rfc.fit(X_train,Y_train)

Y_predict=rfc.predict(X_test)

acc_rfc=round(rfc.score(X_train,Y_train)*100,2)

acc_rfc
# Tabulating the values

models = pd.DataFrame({

    'Model':['Logistic_Regression','Support_Vector_Machine','KNN','GaussianNB','Random_Forest_Classifier'],

    'Score':[acc_log, acc_svc, acc_knn, acc_gaussian, acc_rfc]})

models.sort_values(by='Score',ascending=False)
submission=pd.DataFrame({

            "PassengerId":test_df['PassengerId'],

            "Survived":Y_predict})

submission.head()
filename='submission.csv'

submission.to_csv(filename,index=False)