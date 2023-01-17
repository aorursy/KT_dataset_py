import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.columns)
print(test.columns)
train.isnull().sum()
test.isnull().sum()
train.describe()
sns.countplot(x='Sex',data=train)

print("No of males:"+str((train['Sex']=='male').sum()) )

print("No of females:"+str((train['Sex']=='female').sum()))
sns.barplot(x='Sex',y='Survived',data=train)
#checking what are the different passenger classes.

train.Pclass.unique()
#printing out the no. of people in each class.

sns.countplot(x='Pclass',data=train)

print("No of passengers in 1st class:" + str((train['Pclass']==1).sum()))

print("No of passengers in 2nd class:" + str((train['Pclass']==2).sum()))

print("No of passengers in 3rd class:" + str((train['Pclass']==3).sum()))
#we'll find the ratio of the people in each classes.

sns.barplot(x='Pclass',y='Survived',data=train)
#checking the unique values in siblings and spouse column

train.SibSp.unique()
#printing out the no. of people who boarded the ship with how many siblings and their spouse.

sns.countplot(x='SibSp',data=train)

print("People with no of siblings 0:  " + str((train['SibSp']==0).sum()))

print("People with no of siblings 1:  " + str((train['SibSp']==1).sum()))

print("People with no of siblings 2:  " + str((train['SibSp']==2).sum()))

print("People with no of siblings 3:  " + str((train['SibSp']==3).sum()))

print("People with no of siblings 4:  " + str((train['SibSp']==4).sum()))

print("People with no of siblings 5:  " + str((train['SibSp']==5).sum()))

print("People with no of siblings 8:  " + str((train['SibSp']==8).sum()))
sns.barplot(x='SibSp',y='Survived',data=train)
#Parent child unique values

train.Parch.unique()
sns.countplot(x='Parch',data=train)

print("People who came with 0 Parent or children: "+ str((train['Parch']==0).sum()) )

print("People who came with 1 Parent or children: "+ str((train['Parch']==1).sum()) )

print("People who came with 2 Parent or children: "+ str((train['Parch']==2).sum()) )

print("People who came with 3 Parent or children: "+ str((train['Parch']==3).sum()) )

print("People who came with 4 Parent or children: "+ str((train['Parch']==4).sum()) )

print("People who came with 5 Parent or children: "+ str((train['Parch']==5).sum()) )

print("People who came with 6 Parent or children: "+ str((train['Parch']==6).sum()) )
sns.barplot(x='Parch',y='Survived',data=train)
#Created a new dataframe with NaN values filled in as 0

train_age=train

train_age['Age'].fillna(0)

train_age.head()
train.Age.plot(kind='hist',figsize=[20,7])
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
train.drop(labels=['Name','Ticket','Fare','Cabin'],axis=1,inplace=True)

test.drop(labels=['Name','Ticket','Fare','Cabin'],axis=1,inplace=True)
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()  
#Filling missing values in Embarked 

train['Embarked'].value_counts()
test['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)
train.isnull().sum()
mean_age=np.mean(train['Age'])

mean_age
train['Age'].fillna(mean_age,inplace=True)
train.isnull().sum()
mean_age_test = np.mean(test['Age'])

mean_age_test
test['Age'].fillna(mean_age_test,inplace=True)
test.isnull().sum()
#Map all values to strings

sex_mapping = {'male':0,'female':1}

train['Sex']=train['Sex'].map(sex_mapping)

test['Sex']=test['Sex'].map(sex_mapping)
emb_mapping = {'S':1,'C':2,'Q':3}

train['Embarked']=train['Embarked'].map(emb_mapping)

test['Embarked']=test['Embarked'].map(emb_mapping)
#splitting the data

from sklearn.model_selection import train_test_split
X=train.drop(labels=['PassengerId','Survived'],axis=1)

y=train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)
#K Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
pred_KNN = KNN.predict(X_test)
from sklearn.metrics import accuracy_score
acc_KNN= accuracy_score(y_test,pred_KNN)*100

acc_KNN
#Using Logistic Regression

from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()
LogR.fit(X_train,y_train)
pred_LogR=LogR.predict(X_test)
acc_LogR=accuracy_score(y_test,pred_LogR)*100

acc_LogR
#Decision Tree

from sklearn.tree import DecisionTreeClassifier
Dec_Tree = DecisionTreeClassifier()
Dec_Tree.fit(X_train,y_train)
pred_dec_tree = Dec_Tree.predict(X_test)
acc_dec_tree=accuracy_score(y_test,pred_dec_tree)*100

acc_dec_tree
#Random Forest

from sklearn.ensemble import RandomForestClassifier
RandF = RandomForestClassifier()
RandF.fit(X_train,y_train)
pred_RandF=RandF.predict(X_test)
acc_RandF=accuracy_score(y_test,pred_RandF)*100

acc_RandF
print('Accuracy of K Nearest Neighbours:' + str(acc_KNN))

print('Accuracy of Logistic Regression:' + str(acc_LogR))

print('Accuracy of Decision Tree:' + str(acc_dec_tree))

print('Accuracy of Random Forest :' + str(acc_RandF))
id = test['PassengerId']

predictions = RandF.predict(test.drop(['PassengerId'],axis=1))
output = pd.DataFrame({'PassengerIds':id,'Survived':predictions},index=None)
output.to_csv('submissions.csv',index=False)