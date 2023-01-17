# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe(include= 'all')
print(train.columns)
train.head()
df=pd.DataFrame(train)

df.isnull().sum()
sns.barplot(df['Sex'],df['Survived'])

print('Percentage of females surviving',df['Survived'][df['Sex']=='female'].value_counts(normalize=True)[1]*100)

print('Percentage of males surviving',df['Survived'][df['Sex']=='male'].value_counts(normalize=True)[1]*100)

print('So likely females are going to survive')
sns.barplot(df['Pclass'],df['Survived'])



print('Percentage of pclass = 1 survived',df['Survived'][df['Pclass']==1].value_counts(normalize =True)[1]*100)

print('Percentage of pclass = 2 survived',df['Survived'][df['Pclass']==2].value_counts(normalize =True)[1]*100)

print('Percentage of pclass = 3 survived',df['Survived'][df['Pclass']==3].value_counts(normalize =True)[1]*100)
sns.barplot(df['SibSp'],df['Survived'])

print('Percentage of SibSp = 0 survived',df['Survived'][df['SibSp']==0].value_counts(normalize=True)[1]*100)

print('Percentage of SibSp = 1 survived',df['Survived'][df['SibSp']==1].value_counts(normalize=True)[1]*100)

print('Percentage of SibSp = 2 survived',df['Survived'][df['SibSp']==2].value_counts(normalize=True)[1]*100)

print('Percentage of SibSp = 3 survived',df['Survived'][df['SibSp']==3].value_counts(normalize=True)[1]*100)

print('Percentage of SibSp = 4 survived',df['Survived'][df['SibSp']==4].value_counts(normalize=True)[1]*100)
sns.barplot(df['Parch'],df['Survived'])

print('Percentage of parch = 0 survived',df['Survived'][df['Parch']==0].value_counts(normalize=True)[1]*100)

print('Percentage of parch = 1 survived',df['Survived'][df['Parch']==1].value_counts(normalize=True)[1]*100)

print('Percentage of parch = 2 survived',df['Survived'][df['Parch']==2].value_counts(normalize=True)[1]*100)

print('Percentage of parch = 3 survived',df['Survived'][df['Parch']==3].value_counts(normalize=True)[1]*100)

print('Percentage of parch = 5 survived',df['Survived'][df['Parch']==5].value_counts(normalize=True)[1]*100)

df['Age']= df['Age'].fillna(-0.5)

test['Age']=test['Age'].fillna(-0.5)

bins=[-1,0,5,12,18,24,35,60,np.inf]

labels=['unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Aged']

df['AgeGroup']=pd.cut(df['Age'],bins,labels=labels)

test['AgeGroup']=pd.cut(test['Age'],bins,labels=labels)

plt.figure(figsize=(20,10))

sns.barplot(df['AgeGroup'],df['Survived'])

df['Cabinb']=(df['Cabin'].notnull().astype('int'))

test['Cabinb']=(test['Cabin'].notnull().astype('int'))



sns.barplot(df['Cabinb'],df['Survived'])



print('Percentage of cabin = 0',df['Survived'][df['Cabinb']==0].value_counts(normalize= True)[1]*100)

print('Percentage of cabin = 1',df['Survived'][df['Cabinb']==1].value_counts(normalize= True)[1]*100)
df= df.drop(['Ticket'], axis=1)
df=df.drop(['Cabin'], axis=1)
test=test.drop(['Ticket'],axis=1)

test=test.drop(['Cabin'],axis=1)

print('No of people embarking in Southampton : ')

s=df[df['Embarked']=='S'].shape[0]

print(s)

print('No of people embarking in cherbourg : ')

s=df[df['Embarked']=='C'].shape[0]

print(s)

print('No of people embarking in queenstown : ')

s=df[df['Embarked']=='Q'].shape[0]

print(s)

df=df.fillna({'Embarked':'S'})
df['Embarked'].isnull().any()

for data in df,test:

    data['Title']=data['Name'].str.extract(r'([A-Za-z]+)\.',expand=False)

pd.crosstab(df['Title'], df['Sex'])
for data in df,test:

    data['Title']=data['Title'].replace(['Lady','Capt','Col','Don','Dr','Rev','Major','Jonkheer','Dona'],'Important')

    data['Title']=data['Title'].replace(['Countess','Sir'],'Royal')

    data['Title']=data['Title'].replace(['Mlle','Ms'],'Miss')

    data['Title']=data['Title'].replace(['Mme'],'Mrs')



df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_map={"Mr":1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Important':6}

for data in df,test:

    data['Title']=data['Title'].map(title_map)

    data['Title']=data['Title'].fillna(0)

    

df.head()
df=df.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)
df=df.drop(['Fare'],axis=1)

test=test.drop(['Fare'],axis=1)
sex_map={'male':0,'female':1}

for data in df,test:

    data['Sex']=data['Sex'].map(sex_map)

df.head()
embark_map={'S':1,'C':2,'Q':3}

for data in df,test:

    data['Embarked']=data['Embarked'].map(embark_map)

df.head()
age_map={'Baby':1,'Child':2,'Teenager':3,'Student':4,'Young Adult':5,'Adult':6,'Aged':7,'unknown' : 8}



for data in df,test:

    data['AgeGroup']=data['AgeGroup'].map(age_map)

df.head()



df=df.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)
df.head(6)
test.head()
#TRAINING THE MODEL

df.isnull().any()
from sklearn.model_selection import train_test_split

X=df.drop(['Survived','PassengerId'],axis=1)

Y=df['Survived']



X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
#using logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression().fit(X_train,y_train)

y_pred= lr.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print('The accuracy using logistic Regression is : ',round(accuracy*100,2))
#using SVM



from sklearn.svm import SVC

svc=SVC().fit(X_train,y_train)

y_pred= svc.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print('The accuracy using SVM is : ',round(accuracy*100,2))
#using neural network



from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=[10,10,10],solver='lbfgs',random_state=1).fit(X_train,y_train)

y_pred=mlp.predict(X_test)

accuracy= accuracy_score(y_test,y_pred)

print('The accuracy using neural network is : ',round(accuracy*100,2))

from sklearn.ensemble import RandomForestClassifier

clf =RandomForestClassifier(n_estimators=1700,max_depth=4,max_features='auto',random_state=0).fit(X_train,y_train)

y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print('The accuracy using Random Forest is :',round(accuracy*100,2))
ids= test['PassengerId']

test=test.drop(['PassengerId'],axis=1)


predictions=clf.predict(test)



output= pd.DataFrame({'PassengerId' : ids,'Survived':predictions})

output.to_csv('Submission.csv',index=False)