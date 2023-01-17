import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data=pd.read_csv("../input/train.csv")
train_data.head()
test_data=pd.read_csv("../input/test.csv")
test_data
train_data.info()
train_data.describe()
combo=[train_data,test_data]
for i in combo:

    i['grp size']=1+i['SibSp']+i["Parch"]



print(combo[0].head())
train_data=train_data.drop(['SibSp','Parch'],axis=1)

test_data=test_data.drop(['SibSp','Parch'],axis=1)

combo=[train_data,test_data]
train_data.info()
train_data.describe()
train_data["Pclass"].mode()
train_data["grp size"].mode()
train_data["Pclass"].value_counts()/891
train_data.head()
train_data["grp size"].value_counts()
train_data["Embarked"].value_counts()
train_data.isna().sum()/891
test_data.isna().sum()/891
train_data=train_data.drop(['Cabin'],axis=1)

test_data=test_data.drop(['Cabin'],axis=1)

combo=[train_data,test_data]
train_data.head()
train_data['Age'].mean()
train_data["Embarked"].value_counts()
values={'Age' :29,'Embarked' : 'S'}

train_data=train_data.fillna(value=values)

test_data=test_data.fillna(value=values)

combo=[train_data,test_data]
train_data.isna().sum()
test_data.isna().sum()
train_data.head()
for i in combo:

    i["Embarked"].replace({'S' : 1, 'Q' : 2, 'C' : 3},inplace=True)
train_data.head()
for i in combo:

    i["Sex"].replace({'male' : 0, 'female' : 1},inplace=True)
train_data.head()
train_data=train_data.drop(['Ticket'],axis=1)

test_data=test_data.drop(['Ticket'],axis=1)

combo= [train_data,test_data]
train_data["Age"].value_counts()
train_data["Fare"].value_counts()
"""

0: TODLER

1: KIDS/TEENAGERS

2: ADULTS

3: ELDERLY

"""

for i in combo:

    i.loc[i.Age <= 4, 'age cat'] = 0

    i.loc[(i.Age <=19) & (i.Age>4) , 'age cat'] = 1

    i.loc[(i.Age <= 45) & (i.Age> 19) , 'age cat'] = 2

    i.loc[i.Age>45 , 'age cat'] = 3



test_data["age cat"]=test_data["age cat"].astype('int')

train_data["age cat"]=train_data["age cat"].astype('int')

train_data.head()
train_data.info()
test_data.head()
train_data=train_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Name'],axis=1)
train_data=train_data.drop(['Age','PassengerId'],axis=1)

test_data=test_data.drop(['Age','PassengerId'],axis=1)
combo=[train_data,test_data]
combo[0]
combo[1]
train_data.head()
train_data.hist(figsize=(25,25))

plt.show()
ax = sns.countplot(x = 'grp size', hue = 'Survived', data = train_data)

ax.set( xlabel = 'group size', ylabel = 'Total')
ax = sns.countplot(x = 'Sex', hue = 'Survived', data = train_data)

ax.set( xlabel = 'Sex', ylabel = 'Total')
ax = sns.countplot(x = 'Pclass', hue = 'Survived', data = train_data)

ax.set( xlabel = 'Passenger Class', ylabel = 'Total')
ax =sns.swarmplot(y="age cat", x="Pclass",hue="Survived", data=train_data)

ax.set( ylabel = 'age category', xlabel = 'Class')
sns.swarmplot(y="Sex", x="Pclass",hue="Survived", data=train_data)
sns.swarmplot(y="Fare", x="Pclass",hue="Survived", data=train_data)
train_data.corr()
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data['FareBand'].value_counts()
train_data["Fare"].mode()
values={'Fare' : 8.05}

train_data=train_data.fillna(value=values)

test_data=test_data.fillna(value=values)

combo=[train_data,test_data]

test_data
"""

FARE CATEGORIES

0: (-0.001, 7.91]

1:  (7.91, 14.454]  

2: (14.454, 31.0]  

3: (31.0, 512.329] 

"""

for i in combo:

    i.loc[(i.Fare <= 8.0), 'farecat'] = 0

    i.loc[(i.Fare >8.0) & (i.Fare<=14.5) , 'farecat'] = 1

    i.loc[(i.Fare >14.5) & (i.Fare<=31.0) , 'farecat'] = 2

    i.loc[i.Fare>31.0 , 'farecat'] = 3



train_data.head(20)
train_data["farecat"]=train_data["farecat"].astype('int')

test_data["farecat"]=test_data["farecat"].astype('int')
test_data
train_data=train_data.drop(['Fare','FareBand'],axis=1)

test_data=test_data.drop(['Fare'],axis=1)

combo=[train_data,test_data]
for i in combo:

    print(i)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
X_train = train_data.drop("Survived", axis=1)

y_train = train_data["Survived"]

X_test  = test_data
X_test.head()
X_train.head()
y_train.head()
linreg= LinearRegression()

linreg.fit(X_train,y_train)

Y_pred = linreg.predict(X_test)

linreg.score(X_train, y_train) 
Y_pred
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, y_train) 
knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

Y_pred=knn.predict(X_test)

knn.score(X_train,y_train)
nvbay=GaussianNB()

nvbay.fit(X_train,y_train)

Y_pred=nvbay.predict(X_test)

nvbay.score(X_train,y_train)
dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

Y_pred=dt.predict(X_test)

dt.score(X_train,y_train)
rf=RandomForestClassifier()

rf.fit(X_train,y_train)

Y_pred=rf.predict(X_test)

rf.score(X_train,y_train)
Y_pred=dt.predict(X_test)
test_data=pd.read_csv("../input/test.csv")
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"],"Survived": Y_pred})
submission.to_csv('submission.csv', index=False)