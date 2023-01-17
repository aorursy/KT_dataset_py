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
df = pd.read_csv('../input/train.csv')

df.head()
df.columns
d = df[['Survived','Pclass','Sex', 'Age', 'SibSp',

       'Parch','Embarked']]

d = d.dropna()
from sklearn import preprocessing 
df.columns
import seaborn as sns

import matplotlib.pyplot as plt
grid = sns.FacetGrid(df,col = 'Survived')

grid.map(plt.hist, 'Age',bins = 20)
grid = sns.FacetGrid(df, col = 'Survived', row = "Pclass", size = 1.6, aspect = 1.2)

grid.map(plt.hist, 'Age',bins = 30)

grid.add_legend()
grid = sns.FacetGrid(df, col = 'Survived', row = "Sex", size = 1.6, aspect = 1.2)

grid.map(plt.hist, 'Age',bins = 30)

grid.add_legend() 
grid = sns.FacetGrid(df, col = 'Survived', row = "Embarked", size = 3, aspect = 1.2)

grid.map(plt.hist, 'Age',bins = 30)

grid.add_legend()
grid = sns.FacetGrid(df, row = 'Embarked', size = 3.2, aspect = 1.2)

grid.map(sns.pointplot, 'Pclass','Survived','Sex',palette = 'deep')

grid.add_legend()
df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\. ', expand = False)

pd.crosstab(df.title,df.Sex)
df['title'] = df.title.replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



df['title'] = df['title'].replace('Mlle', 'Miss')

df['title'] = df['title'].replace('Ms', 'Miss')

df['title'] = df['title'].replace('Mme', 'Mrs')

df[['title','Survived']].groupby('title').mean()
df.title = df.title.fillna(0)

df.title = df.title.fillna(0)

df['title'] = df.title.map({'Rare':0, 'Master':1, 'Miss':2, 'Mr':3, 'Mrs':4})

df.Sex = df.Sex.map({'female':0, 'male':1}).astype(int)
df.info()
df['Age'].shape
df.Age = df.Age.fillna(df.Age.mean())

df.Embarked = df.Embarked.fillna('S')
df = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

df['Embarked'] = df.Embarked.map({'S':0,'C':1,'Q':2}).astype(int)

df.Age = round(df.Age).astype(int)

df.head()

df['AgeBand'] = pd.cut(df.Age,5)

df[['AgeBand','Survived']].groupby('AgeBand',as_index = False).mean()
df.loc[(round(df['Age'])<=16),'Age'] = 0

df.loc[(round(df['Age'])>16) & (round(df['Age'])<=32),'Age'] = 1

df.loc[(round(df['Age'])>32) & (round(df['Age'])<=48),'Age'] = 2

df.loc[(round(df['Age'])>48) & (round(df['Age'])<=64),'Age'] = 3

df.loc[(round(df['Age'])>64) & (round(df['Age'])<=80),'Age'] = 4
df[['Age','Survived']].groupby('Age',as_index = False).mean()
df.head()
df['Fare'] = df.Fare.fillna(df.Fare.mean())

df['FareBand'] = pd.qcut(df.Fare,4)

df[['FareBand','Survived']].groupby('FareBand',as_index = False).count().sort_values(by = 'FareBand',ascending = True)

#df['FareBand']
df.loc[(round(df['Fare'])<=7.9),'Fare'] = 0

df.loc[(round(df['Fare'])>7.9) & (round(df['Fare'])<=14.45),'Fare'] = 1

df.loc[(round(df['Fare'])>14.45) & (round(df['Fare'])<=31.0),'Fare'] = 2

df.loc[(round(df['Fare'])>31.0),'Fare'] = 3
df[['Fare','Survived']].groupby('Fare').mean()
df.drop(columns = ['AgeBand','FareBand']).head()
df['Familysize'] = df['SibSp'] + df['Parch'] + 1

df[['Familysize','Survived']].groupby('Familysize').mean()
df['Familysize'] = df['SibSp'] + df['Parch'] + 1

df.loc[df['Familysize'] == 1,'Familysize'] = 0

df.loc[df.Familysize >1,'Familysize'] = 1
df[['Familysize','Survived']].groupby('Familysize').mean()
df = df.drop(columns = ['Parch','SibSp'])
df = df.drop(columns = ['AgeBand','FareBand'])

df.head()
#df['Embarked'] = df.Embarked.map({'S':0,'C' :1, 'Q': 2})

df.Embarked = df.Embarked.fillna(0)

df['Embarked'] = round(df['Embarked']).astype(int)

df.Fare = df.Fare.astype(int)

df.info()
df.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
DTC = DecisionTreeClassifier()

RFC = RandomForestClassifier(n_estimators = 500)

LR = LogisticRegression()

GNB = GaussianNB()

MLPC = MLPClassifier()

svc = SVC(kernel = 'linear', C = 0.1, gamma = 'scale')
X = np.asanyarray(df.drop(columns = ['Survived']))

y = np.asanyarray(df.Survived)

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(X,y , test_size = 0.2,random_state = 0)
DTC.fit(x_train,y_train)

y_hat1 = DTC.predict(x_test)

RFC.fit(x_train,y_train)

y_hat2 = RFC.predict(x_test)

LR.fit(x_train,y_train)

y_hat3 = LR.predict(x_test)

GNB.fit(x_train,y_train)

y_hat4 = GNB.predict(x_test)

MLPC.fit(x_train,y_train)

y_hat5 = MLPC.predict(x_test)

svc.fit(x_train,y_train)

y_hat6 = svc.predict(x_test)
from sklearn.metrics import classification_report, accuracy_score

print('Accuracy_yhat1_DTC=',accuracy_score(y_test,y_hat1))

print('Accuracy_yhat2_RFC=',accuracy_score(y_test,y_hat2))

print('Accuracy_yhat3_LR=',accuracy_score(y_test,y_hat3))

print('Accuracy_yhat4_GNB=',accuracy_score(y_test,y_hat4))

print('Accuracy_yhat5_MLPC=',accuracy_score(y_test,y_hat5))

print('Accuracy_yhat6_SVC=',accuracy_score(y_test,y_hat6))
DTC.score(x_train,y_train)
RFC.score(x_train,y_train)
from sklearn.ensemble import ExtraTreesClassifier

ETC = ExtraTreesClassifier(n_estimators = 500)

ETC.fit(x_train,y_train)

y_hat7 = ETC.predict(x_test)

ETC.score(x_train,y_train)

print('Accuracy_yhat7_ETC=',accuracy_score(y_test,y_hat7))
df1 = pd.read_csv('../input/test.csv')

df1.head()
df1['title'] = df1['Name'].str.extract(' ([A-Za-z]+)\. ', expand = False)

df1['title'] = df1.title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



df1['title'] = df1['title'].replace('Mlle', 'Miss')

df1['title'] = df1['title'].replace('Ms', 'Miss')

df1['title'] = df1['title'].replace('Mme', 'Mrs')
df1.title = df1.title.fillna(0)

df1['title'] = df1.title.map({'Rare':0, 'Master':1, 'Miss':2, 'Mr':3, 'Mrs':4})

df1.Sex = df1.Sex.map({'female':0, 'male':1}).astype(int)
df1 = df1.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

df1['Embarked'] = df1.Embarked.map({'S':0,'C':1,'Q':2}).astype(int)

df1.Age = df1.Age.fillna(df1.Age.mean())



df1.Age = round(df1.Age).astype(int)

df1.loc[(round(df1['Age'])<=16),'Age'] = 0

df1.loc[(round(df1['Age'])>16) & (round(df1['Age'])<=32),'Age'] = 1

df1.loc[(round(df1['Age'])>32) & (round(df1['Age'])<=48),'Age'] = 2

df1.loc[(round(df1['Age'])>48) & (round(df1['Age'])<=64),'Age'] = 3

df1.loc[(round(df1['Age'])>64) & (round(df1['Age'])<=80),'Age'] = 4
df1.loc[(round(df1['Fare'])<=7.9),'Fare'] = 0

df1.loc[(round(df1['Fare'])>7.9) & (round(df1['Fare'])<=14.45),'Fare'] = 1

df1.loc[(round(df1['Fare'])>14.45) & (round(df1['Fare'])<=31.0),'Fare'] = 2

df1.loc[(round(df1['Fare'])>31.0),'Fare'] = 3
df1['Familysize'] = df1['SibSp'] + df1['Parch'] + 1

df1.loc[df1['Familysize'] == 1,'Familysize'] = 0

df1.loc[df1.Familysize >1,'Familysize'] = 1
df1 = df1.drop(columns = ['Parch','SibSp'])

df1.Fare = df1.Fare.fillna(0)

df1.head()
predictions = RFC.predict(np.asanyarray(df1))
df3 = pd.read_csv('../input/test.csv')

submissions = pd.DataFrame({'PassengerID':df3['PassengerId'],'Survived':predictions})

submissions.to_csv('submission.csv',index = False, header = True)