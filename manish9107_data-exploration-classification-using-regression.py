# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd                    # For Data Exploration

import numpy as np                     # For mathematical calculations 

import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 

import warnings                        # To ignore any warnings 

warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/train.csv')

train_df.columns
train_df.head(5)
test_df = pd.read_csv('../input/test.csv')

test_df.columns
test_df.head(5)
train_original = train_df.copy()

test_original = test_df.copy()
train_df.dtypes
train_df.shape, test_df.shape
train_df.isnull().sum()
train_df['Age'].fillna(train_df['Age'].mean(),inplace = True)
train_df['Cabin'].fillna(train_df['Cabin'].mode()[0], inplace = True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df.isnull().sum()
test_df['Age'].fillna(test_df['Age'].mean(),inplace = True)

test_df['Cabin'].fillna(test_df['Cabin'].mode()[0], inplace = True)
train_df['Survived'].value_counts(normalize = True)
train_df['Survived'].value_counts(normalize = True).plot.bar(title = "Survival %")
train_df['Pclass'].value_counts(normalize =  True)
train_df['Sex'].value_counts(normalize= True)
train_df['Embarked'].value_counts()
train_df['SibSp'].value_counts()
train_df['Parch'].value_counts()
plt.figure(1)

plt.subplot(121) 

sns.distplot(train_df['PassengerId']); 

plt.subplot(122) 

train_df['PassengerId'].plot.box(figsize=(16,5)) 

plt.show()
plt.figure(2)

plt.subplot(221)

df = train_df.dropna()

sns.distplot(df['Age']);

plt.subplot(222)

df['Age'].plot.box(figsize = (16,5))

plt.show()
plt.figure(3)

plt.subplot(321)

sns.distplot(train_df['Fare']);

plt.subplot(322)

train_df['Fare'].plot.box(figsize=(16,5))

plt.show()
Ticket_Class=pd.crosstab(train_df['Pclass'],train_df['Survived'])

Sex=pd.crosstab(train_df['Sex'],train_df['Survived'])

Siblings=pd.crosstab(train_df['SibSp'],train_df['Survived'])

Parents=pd.crosstab(train_df['Parch'],train_df['Survived'])

Embarked=pd.crosstab(train_df['Embarked'],train_df['Survived'])
Ticket_Class.div(Ticket_Class.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.show() 

Sex.div(Sex.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.show() 

Siblings.div(Siblings.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.show() 

Parents.div(Parents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.show() 

Embarked.div(Embarked.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.show() 
train_df.groupby('Survived')['PassengerId'].mean().plot.bar()

plt.show()
bins=[0,12,20,60,80] 

group=['Children','Teenage','Adult', 'Senior Citizen'] 

train_df['Age_bin']=pd.cut(df['Age'],bins,labels=group)

Age_bin=pd.crosstab(train_df['Age_bin'],train_df['Survived']) 

Age_bin.div(Age_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('Age') 

P = plt.ylabel('Percentage')

plt.show()
matrix = train_df.corr() 

fax = plt.subplots(figsize=(9, 6)) 

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train = train_df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Age_bin'], axis =1)

test = test_df.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis =1)
X = train.drop('Survived',1)

y = train.Survived
X=pd.get_dummies(X) 

train=pd.get_dummies(train) 

test=pd.get_dummies(test)
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score

model = LogisticRegression() 

model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
pred_test = model.predict(test)
Passenger_Id = test_df['PassengerId'].values
gender_submission = pd.DataFrame({'PassengerId': Passenger_Id, 'Survived': pred_test}, columns=['PassengerId', 'Survived'])
gender_submission.to_csv('gender_submission.csv', index = False)