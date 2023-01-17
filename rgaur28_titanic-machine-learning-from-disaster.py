# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing libraries necessary for the study

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# reading the train dataset

train = pd.DataFrame(pd.read_csv('/kaggle/input/titanic/train.csv'))

train.head() 
# reading the test dataset

test = pd.DataFrame(pd.read_csv('/kaggle/input/titanic/test.csv'))

test.head() 
#Creating a copy 

test_copy=test.copy()
train.shape
test.shape
train.describe()
train.dtypes
round(100*(train.isnull().sum()/len(train.index)), 2).sort_values(ascending = False) 
#Cabin has 77% null data hence we can drop this column

train.drop('Cabin',axis=1,inplace=True)
#Cabin has 77% null data hence we can drop this column

test.drop('Cabin',axis=1,inplace=True)
#remove null values in the dataset

train.dropna(inplace=True)
#unique value in dataframe

cols = train.columns

for i in cols:

    print(i,train[i].unique(),'\n','*********************************************')
train.groupby('Pclass').Survived.describe()
train.groupby('Sex').Survived.describe()
#label encoding

train['Sex']=train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#label encoding

test['Sex']=test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.groupby('SibSp').Survived.describe()
train.groupby('Parch').Survived.describe()
train.head()
#Remove columns

train.drop(['Name','Ticket','Fare','PassengerId'],axis=1,inplace=True)
test.drop(['Name','Ticket','Fare'],axis=1,inplace=True)
#Bin the column

train['Age_Group']=0

train.loc[(train['Age']<=15) ,'Age_Group']= 0

train.loc[(train['Age']>15) & (train['Age']<=30) ,'Age_Group']=1

train.loc[(train['Age']>30) & (train['Age']<=45), 'Age_Group']=2

train.loc[(train['Age']>45) & (train['Age']<=60) ,'Age_Group']=3

train.loc[(train['Age']>60) & (train['Age']<=75) ,'Age_Group']=4

train.loc[(train['Age']>75),'Age_Group']=5
train.Age.hist(bins=20);
test['Age_Group']=0

test.loc[(test['Age']<=15) ,'Age_Group']= 0

test.loc[(test['Age']>15) & (test['Age']<=30) ,'Age_Group']=1

test.loc[(test['Age']>30) & (test['Age']<=45), 'Age_Group']=2

test.loc[(test['Age']>45) & (test['Age']<=60) ,'Age_Group']=3

test.loc[(test['Age']>60) & (test['Age']<=75) ,'Age_Group']=4

test.loc[(test['Age']>75),'Age_Group']=5
train.groupby('Embarked').Survived.describe()
#label encoding

train['Embarked']=train['Embarked'].map( {'S': 1, 'C': 0,'Q':2} ).astype(int)
#label encoding

test['Embarked']=test['Embarked'].map( {'S': 1, 'C': 0,'Q':2} ).astype(int)
train.head()
test.head()
plt.figure(figsize=(16, 6))

sns.barplot(x="Sex", y="Survived", hue="Age_Group", data=train ,palette="deep")

plt.ylabel('')

plt.show()
import matplotlib.pyplot as plt

gd = sns.FacetGrid(train, col="Survived",  row="Pclass")

gd = gd.map(plt.hist, "Age",color="c")
import matplotlib.pyplot as plt

gd = sns.FacetGrid(train, col="Survived",  row="Embarked")

gd = gd.map(plt.hist, "Age",color="c")
import matplotlib.pyplot as plt

gd = sns.FacetGrid(train, col="Survived",  row="Sex")

gd = gd.map(plt.hist, "Age",color="c")
train.drop(['Age'],axis=1,inplace=True)
test.drop(['Age'],axis=1,inplace=True)
#heatmap

plt.figure(figsize=(25,25))

sns.heatmap(train.corr(), annot= True)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_train.shape, Y_train.shape
X_test  = test.drop("PassengerId", axis=1).copy()

X_test.shape
X_train.columns
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)

y_pred = log_reg.predict(X_test)

round(log_reg.score(X_train, Y_train) * 100, 2)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

y_pred = decision_tree.predict(X_test)

round(decision_tree.score(X_train, Y_train) * 100, 2)
from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(n_estimators=100)

ada_boost.fit(X_train, Y_train)

y_pred = ada_boost.predict(X_test)

ada_boost.score(X_train, Y_train)

round(ada_boost.score(X_train, Y_train) * 100, 2)
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier(n_estimators=100)

gradient_boost.fit(X_train, Y_train)

y_pred = gradient_boost.predict(X_test)

gradient_boost.score(X_train, Y_train)

round(gradient_boost.score(X_train, Y_train) * 100, 2)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, Y_train)

y_pred = svc.predict(X_test)

round(svc.score(X_train, Y_train) * 100, 2)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

round(random_forest.score(X_train, Y_train) * 100, 2)
sub = pd.DataFrame()

sub['PassengerId'] = test_copy['PassengerId']

sub['Survived'] = y_pred

sub.to_csv('Submission_Titanic.csv',index=False)