import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ss = pd.read_csv('../input/gender_submission.csv')

train.head()
sns.pairplot(train)

train.info()

train.describe()
Survived = train[train['Survived']==1]

Survived
no_Survived = train[train['Survived']==0]

no_Survived
print('Number of Survived passengers = ',len(Survived))   

print('Percentage Survived = ',len(Survived)/len(train)*100,'%')   

print('-----------------')

print('Number of passengers who did not Survive = ',len(no_Survived))

print('Percentage No Survived = ',len(no_Survived)/len(train)*100,'%')



print('-----------------')

print('Total = ',len(train))

 



plt.figure(figsize=[6,12])

plt.subplot(211)

sns.countplot(x='Pclass',data=train)

plt.subplot(212)

sns.countplot(x='Pclass',hue='Survived',data=train)
plt.figure(figsize=[6,12])

plt.subplot(211)

sns.countplot(x='SibSp',data=train)

plt.subplot(212)

sns.countplot(x='SibSp',hue='Survived',data=train)
plt.figure(figsize=(80,30))

sns.countplot(x='Age',hue='Survived',data=train)
train['Age'].hist(bins=40)
train['Fare'].hist(bins=40)
sns.heatmap(train.isnull(),cmap='Blues')
train.drop('Cabin',axis=1,inplace=True) #remove Cabin col

train
sns.heatmap(train.isnull(),cmap='Blues')
plt.figure(figsize=(15,10))

sns.boxplot(x='Sex',y='Age',data=train)
def Fill_Age(data):

    age=data[0]

    sex=data[1]

    if pd.isnull(age):

        if sex is 'male':

            return 29

        else:

            return 25

    else:

        return age   

        
train['Age']=train[['Age','Sex']].apply(Fill_Age,axis=1)
sns.heatmap(train.isnull(),cmap='Blues')
train.drop(['Name','Ticket','Embarked','PassengerId'],axis=1,inplace=True)

train
male=pd.get_dummies(train['Sex'],drop_first=True) # convert male=1 and female=0
male
train.drop(['Sex'],axis=1,inplace=True)
train=pd.concat([train,male],axis=1)  # axis=1 ==>rows   , axis=0 ==>cols
train
X=train.drop('Survived',axis=1).values 

y=train['Survived'].values
X
y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

Classifier=LogisticRegression()

Classifier.fit(X_train,y_train)
y_perdict_test=Classifier.predict(X_test)
y_perdict_test
y_test
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_perdict_test)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_perdict_test))