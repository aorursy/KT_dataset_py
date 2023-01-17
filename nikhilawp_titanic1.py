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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing 

le=preprocessing.LabelEncoder()

from sklearn.preprocessing import StandardScaler
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


test.head()

train.head()
train=train.drop(columns=['Name','Ticket','Cabin'])
train['Sex'].unique()

train['Sex']=le.fit_transform(train['Sex'])

            
train.dropna()
train.set_index('PassengerId',inplace=True)

train=train.dropna()

train=train.drop(columns='Age')
train['Embarked']=le.fit_transform(train['Embarked'])

train
x=StandardScaler().fit_transform(train)

x
from sklearn.linear_model import LogisticRegression
y=train['Survived']

train=train.drop(columns='Survived')
lr=LogisticRegression

test=test.drop(columns=['Name','Ticket','Cabin'])

test['Sex']=le.fit_transform(test['Sex'])

test['Embarked']=le.fit_transform(test['Embarked'])

#rough

test=test.drop(columns='Age')
test['Fare']=test['Fare'].fillna(method='ffill')

test.set_index('PassengerId',inplace=True)
LR1=lr(C=1.02,solver='sag').fit(x,y)
y_test=LR1.predict(test)



sub1=pd.DataFrame(y_test)

sub1=sub1.set_index(test.index)

sub1=sub1.rename(columns={0:"Survived"})
sub1.to_csv('sub1.csv')
sub1['Survived'].value_counts()