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
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv' )
full_data = [train, test]

print (train.info())
#Columns with missing values: Age,Cabin,Embarked
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()) 

print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1  #total members of family
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')      #filling missing values
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())  #filling missing values
train['grpFare'] = pd.qcut(train['Fare'], 4)                          #creating grpFare size of 4 
print (train[['grpFare', 'Survived']].groupby(['grpFare'], as_index=False).mean())
for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(train['Age'].median())     #filling missing values
train['grpAge'] = pd.cut(train['Age'], 5)                             #making 5 ageGrps of same age diffenerence
print (train[['grpAge', 'Survived']].groupby(['grpAge'], as_index=False).mean())
fig, saxis = plt.subplots(2,3,figsize=(24,16))

sns.barplot(x = 'Embarked', y = 'Survived', data=train,ax = saxis[0,0])
sns.pointplot(x = 'grpFare', y = 'Survived', data=train,ax = saxis[0,1])
sns.barplot(x = 'Sex', y = 'Survived', data=train,ax = saxis[0,2])
sns.barplot(x = 'Pclass', y = 'Survived', data=train,ax = saxis[1,0])
sns.barplot(x = 'IsAlone', y = 'Survived', data=train,ax = saxis[1,1])
sns.barplot(x = 'grpAge', y = 'Survived', data=train,ax = saxis[1,2])