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
%matplotlib inline





from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import xgboost as xgb

from multiprocessing import Pool
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
## analyse the data 

gender_submission.info()

print("--------------------------------------------------")

test.info()

print("--------------------------------------------------")

train.info()
## initial anysis says that this is regression issue as  features can increaSE in y. And there lot of dependency in the data  



#lets dig deeper in the data 

train.describe(include='all')


# %% [code]

##grouping the data 

sns.set()

analysis1=train.groupby(['Sex'])[['Survived']].mean()

print(analysis1)

analysis1.plot(kind='bar',stacked=True)



# %% [code]

analysis2=train.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack()

print(analysis2)

analysis2.plot(kind='bar')



# %% [code]

age = pd.cut(train['Age'], [0, 18, 80])

#titanic.pivot_table('survived', index='sex', columns='class')

#  call signature as of Pandas 0.18

# DataFrame.pivot_table(data, values=None, index=None, columns=None,

#                       aggfunc='mean', fill_value=None, margins=False,

#                       dropna=True, margins_name='All')



analysis3=train.pivot_table('Survived', ['Sex', age], 'Pclass')

print(analysis3)

analysis3.plot(kind='bar')



# %% [code]

fare = pd.qcut(train['Fare'], 2)

analysis4=train.pivot_table('Survived', ['Sex', age], [fare, 'Pclass'])

print(analysis4)

analysis4.plot(kind='bar',stacked=True)



# %% [code]



analysis5=train.pivot_table(index='Sex', columns='Pclass',

                    aggfunc={'Survived':sum, 'Fare':'mean'})

print(analysis5)

analysis5.plot(kind='bar')



# %% [code]

analysis6=train.pivot_table('Survived', index='Sex', columns='Pclass', margins=True)

print(analysis6)

analysis6.plot(kind='bar')
## cleaning the data 

train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

train.head()
#Convert  to [1,0] so that our decision tree can be built

for df in [train,test]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})

    

#Fill in missing age values with 0 

train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)



#Select feature column names and target variable we are going to use for training

features = ['Pclass','Age','Sex_binary']

target = 'Survived'





train[features].head()



#Create classifier object 

clf = DecisionTreeClassifier()  

#Fit 

clf.fit(train[features],train[target]) 
predictions = clf.predict(test[features])

predictions


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



submission.head()



filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)