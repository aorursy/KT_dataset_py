#importing libraries

%matplotlib inline

import matplotlib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#load training dataset

train = pd.read_csv("../input/train.csv")

train.head(4)
train.shape
#load testing dataset

test=pd.read_csv("../input/test.csv")

test.head()
test.shape
#store the passenger id in test data

passenger_id=test['PassengerId']
test.head(5)
train.set_index(['PassengerId'],inplace=True)#set index as PassengerId

train.head()
test.set_index(['PassengerId'],inplace=True)#set index as PassengerId

test.head()
train.isnull().sum()#finding missing value in each columns of train data 
test.isnull().sum()#finding missing value in each columns of test data
# Cleaning the data 

#first imputing the missing value for train data

from sklearn.preprocessing import Imputer

im=Imputer(missing_values='NaN',strategy='median',axis=1)

Age2=im.fit_transform(train['Age'].values.reshape(1,-1))

Age2=Age2.T

Age2

train['Age2']=Age2

train.head()
#imputing the missing value for test data

from sklearn.preprocessing import Imputer

im=Imputer(missing_values='NaN',strategy='median',axis=1)

Age2=im.fit_transform(test['Age'].values.reshape(1,-1))

Age2=Age2.T

Age2

test['Age2']=Age2

test.head()
train.isnull().sum() #checking null values in train data
#filling the embarked missing values

train.Embarked.value_counts()
#filling S in embarked missing values

train.Embarked.fillna('S',inplace=True)
train.isnull().sum()
test.isnull().sum()
test.Fare.fillna(test.Fare.mean(),inplace=True)
test.isnull().sum()
#droping age and cabin columns 

train.drop(['Age','Cabin'],axis=1,inplace=True)
train.isnull().sum()
test.drop(['Age','Cabin'],axis=1,inplace=True)
test.isnull().sum()
train.head()
train['Sex']=train.Sex.apply(lambda x:1 if x=='male' else 0) #transforming sex column into 0 and 1

train.Sex.head()
test['Sex']=test.Sex.apply(lambda x:1 if x=='male' else 0) #transforming sex column into 0 and 1

test.head()
train.describe()
#checking dependancy of one variable to another

#gropby survived

train.groupby('Survived').mean()
#groupby sex

train.groupby('Sex').mean()
#checking correalation btw one variable to each other

train.corr()
#creating variable realtion and visulize them

#for sex and survived

sns.barplot(x='Sex',y='Survived',data=train)
#for pclass and survived

sns.barplot(x='Pclass',y='Survived',data=train)
#creating a new feature name family_size by adding sibsp and parch column

train['family_size']=train["SibSp"]+train["Parch"]+1 #add 1 because if there is any who travel alone

test['family_size']=test["SibSp"]+test["Parch"]+1
train.head()
#creatin g a new feature name family_group

def family_group(size):

    a=''

    if (size<=1):

        a='alone'

        

    elif (size<=4):

        a='small'

    else:

        a='large'

    return a

train['family_group']=train.family_size.map(family_group)

test['family_group']=test.family_size.map(family_group)

train.head()
test.head()
#categories by age

def age_group(age):

    a=''

    if (age<=1):

        a='infant'

    elif (age<=4):

        a='toddler'

    elif (age<=12):

        a='child'

    elif (age<=15):

        a='teenager'

    elif (age<=25):

        a='young_adult'

    elif (age<=40):

        a='adult'

    elif (age<=55):

        a='middle_age'

    else:

        a='old'

    return a

        
train['age_group']=train['Age2'].map(age_group)

test['age_group']=test['Age2'].map(age_group)

train.head()
train.age_group.value_counts()

test.age_group.value_counts()
train['fare_per_person']=train['Fare']/train['family_size']

train.head()

test['fare_per_person']=test['Fare']/test['family_size']

test.head()
def fare_group(fare):

    a=''

    if (fare<=4):

        a='very_low'

    elif (fare<=10):

        a='low'

    elif (fare<=20):

        a='mid'

    elif (fare<=45):

        a='high'

    else:

        a='very_high'

    return a

train['fare_group']=train.fare_per_person.map(fare_group)

train.head()

test['fare_group']=test.fare_per_person.map(fare_group)

test.head()
train['fare_group'].value_counts()

test['fare_group'].value_counts()
test.shape
train=pd.get_dummies(train,columns=['Embarked','family_group','age_group','fare_group'],drop_first=True)

test=pd.get_dummies(test,columns=['Embarked','family_group','age_group','fare_group'],drop_first=True)
train.shape
test.shape
train.drop(['Name','Fare','Age2','Ticket','fare_per_person','family_size'],axis=1,inplace=True)

test.drop(['Name','Fare','Age2','Ticket','fare_per_person','family_size'],axis=1,inplace=True)
train.head()
test.head()
train.shape
test.shape
X=train.drop("Survived",axis=1)

Y=train["Survived"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.469,random_state=0)
X_test.shape
#importing models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier
#LogisticRegression

log=LogisticRegression()
log.fit(X_train, Y_train)
Y_pred = log.predict(X_test)
from sklearn.metrics import accuracy_score
log_acc=accuracy_score(Y_test,Y_pred)

log_acc
#Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_test)
rfc_acc=accuracy_score(Y_test,Y_pred)

rfc_acc
from sklearn import svm

#linear svm

svc= svm.SVC(kernel='linear')
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
from sklearn.metrics import accuracy_score

L_svm_acc=accuracy_score(Y_test,Y_pred)

L_svm_acc
#svm

svc1=svm.SVC(kernel='rbf')

svc1.fit(X_train,Y_train)

Y_pred=svc1.predict(X_test)
from sklearn.metrics import accuracy_score

svm_acc=accuracy_score(Y_test,Y_pred)

svm_acc
#KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score

knn_acc=accuracy_score(Y_test,Y_pred)

knn_acc
#GaussianNB

gaus = GaussianNB()
gaus.fit(X_train, Y_train)
Y_pred = gaus.predict(X_test)
from sklearn.metrics import accuracy_score

gaus_acc=accuracy_score(Y_test,Y_pred)

gaus_acc
#Perceptron model

perptr = Perceptron()

perptr.fit(X_train, Y_train)
Y_pred=perptr.predict(X_test)
from sklearn.metrics import accuracy_score

perptr_acc=accuracy_score(Y_test,Y_pred)

perptr_acc
#Decision Tree Classsifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)
from sklearn.metrics import accuracy_score

dt_acc=accuracy_score(Y_test,Y_pred)

dt_acc
output=pd.DataFrame({

'Model' :['LogisticRegression','SVM','LinearSVM','RandomForestClassifier',

          'KNeighborsClassifier','GaussianNB','Perceptron','DecisionTreeClassifier'],

'Accuracy' : [log_acc,svm_acc,L_svm_acc,rfc_acc,knn_acc,gaus_acc,perptr_acc,dt_acc]})
output
#Visualization

ax=plt.subplots(figsize=(10,6))

ax=sns.barplot(x="Accuracy",y="Model",data=output,color='b')

ax.set_xlabel("Accuracy",fontsize=15)

plt.ylabel("Model",fontsize=15)

plt.grid(color='r',linestyle='-',linewidth=0.5)

plt.title("Model Accuracy",fontsize=20)
Submission = pd.DataFrame(pd.DataFrame({

        "PassengerId": passenger_id,

        "Survived": Y_pred

    }))

 #Submission.to_csv('../output/submission.csv', index=False)
#Submission.to_csv('Titanic_Prediction', index=False)