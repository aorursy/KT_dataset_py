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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print(train.info())

print(test.info())
#Extracting the last name of each passenger and storing it in a new column called 'LastName'

x=[]

for i in range(0,891):

    a=train['Name'][i].split(',')[0]

    x.append(a)

x=np.array(x)

train['LastName']=x

print('Unique values in "LastName" column of the training data are: ' + str(train['LastName'].nunique()))

print(train.info())
#'Ticket' column contains an optional Station code and ticket number.

#Extracting Station code and Ticket Number and stroing them in columns 'Station' & 'TicketNumber' respectively.

print(train['Ticket'].head())

Station=[]

TicketNumber=[]

for i in range(0,891):

    a=train['Ticket'][i].split()

    if len(a)==1:

        Station.append(np.nan)

        TicketNumber.append(a[0])

    elif len(a)==2:

        Station.append(a[0])

        TicketNumber.append(a[1])

    else:

        Station.append(a[0])

        TicketNumber.append(a[2])

train['Station']=np.array(Station)

train['TicketNumber']=np.array(TicketNumber)

train['Station'][train['Station']=='nan']=np.NaN

train['TicketNumber'][train['TicketNumber']=='LINE']=-1

train['Station'].fillna(np.NaN,inplace=True)

print(train.info())

print("The number of unique values in station are: " + str(train['Station'].nunique()))
#Checking info about the missing 2 missing values of 'Embarked' column

print(train[train['Embarked'].isnull()==True])

print(train[['Survived','Name','Age']][train['Embarked'].isnull()==True])
#Icard was Mrs.Nelson's maid, this means their embarked port is same. Let's find out!

#A quick google search tells us that their embarkation port was Southampton i.e, 'S'. It also tells us that their cabin was B-28.

train['Embarked'].fillna('S',inplace=True)

print(train.info())
#Extracting Cabin type(A,B,C,D,E,F,G) from Cabin and storing it in Cabin again

print(train['Cabin'].head(10))

train['Cabin']=train['Cabin'].str.extract('([A-Z])',expand=True)

#Fill the missing cabin values with 'U'

train['Cabin'].fillna('U',inplace=True)

print(train['Cabin'].head(10))

print(train.info())
#Histogram and distplot of 'Age'

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

plt.subplot(1,2,1)

plt.hist(train['Age'],bins=10)

plt.xlabel('Age')

plt.ylabel('Count')

plt.subplot(1,2,2)

sns.distplot(train['Age'][train['Age'].isnull()==False],bins=10)

plt.tight_layout()

#The graph is negatively skewed
#Filling the missing 'Age' values with random integers between (mean-std,mean+std) of 'Age'

train['Age'].fillna(np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std()),inplace=True)

#train['Family']=train['Parch']+train['SibSp']+1

print(train.info())
#Encoding columns

to_drop=['Name','Ticket','Station','TicketNumber','PassengerId','LastName']#'Parch','SibSp'

to_int=['TicketNumber']

to_cat=['Sex','Cabin','Embarked','LastName']

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

for col in to_cat:

    train[col]=encoder.fit_transform(train[col])

train['TicketNumber']=train['TicketNumber'].astype(int)

print(train.info())
#Create x_train by dropping the above tables

x_train=train.drop(to_drop,axis=1)

x_train.drop('Survived',axis=1,inplace=True)

y_train=train['Survived']

print(x_train.info())

print(y_train.head())
#Let's visualise our data

cols=['Pclass','Sex','Embarked']#'Family'

for col in cols:

    sns.countplot(train[col],hue=train['Survived'],palette="Set2")

    plt.show()

#Conclusion from these plot - Survival rate of females are much more than male, survival rate is very high in Pclass 3 and for people who embarked from '2'
#Cleaning the test data

test['Cabin']=test['Cabin'].str.extract('([A-Z])',expand=True)

test['Cabin'].fillna('U',inplace=True)

print(test[['Pclass','Cabin','Embarked']][test['Fare'].isnull()==True])

med=test['Fare'][np.logical_and(test['Pclass']==3,np.logical_and(test['Cabin']=='U',test['Embarked']=='S'))].median()

print("The median of Fare where Pclass, Cabin and Embarked features matched is " + str(med))

test['Fare'].fillna(med,inplace=True)

test['Age'].fillna(np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std()),inplace=True)

print(test.info())
x=[]

for i in range(0,418):

    a=test['Name'][i].split(',')[0]

    x.append(a)

x=np.array(x)

test['LastName']=x

#test['Family']=test['Parch']+test['SibSp']+1

for col in to_cat:

    test[col]=encoder.fit_transform(test[col])

print(test.info())

print(test.head())
to_drop=['Name','Ticket','PassengerId','LastName']#'SibSp','Parch'

x_test=test.drop(to_drop,axis=1)

print(x_test.info())
#Tried xgb, LogisticRegression and DecisionTreeClassifier

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

model=xgb.XGBClassifier()#Keep an eye on hyper-parameters

modelL=LogisticRegression()

modelDTC=DecisionTreeClassifier()

model.fit(x_train,y_train)

modelL.fit(x_train,y_train)

modelDTC.fit(x_train,y_train)

print("Score on training data for xgbclassifier is around " + str(int(model.score(x_train,y_train)*100)) + "%")

print("Score on training data for LogisticRegression is around " + str(int(modelL.score(x_train,y_train)*100)) + "%")

print("Score on training data for DecisionTreeClassifier is around " + str(int(modelDTC.score(x_train,y_train)*100)) + "%")
#Going ahead with xgb

y_test=model.predict(x_test)

ans=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_test})

ans.to_csv('outputxgb.csv',index=False)