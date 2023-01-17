# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
gender_sub = pd.read_csv('../input/gender_submission.csv')
test = pd.read_csv('../input/test.csv')
#showing the sample train data 
train.head()
# describe the train dataset
train.describe()
#checking for null values in data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Count of survived and those who don't
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
# Those who survived (male /female)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# survived on basis of class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# column has so much null values
train=train.drop('Cabin',axis=1)
train.head()
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
# Average age and passanger class
plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
# this graph is showing that there is no null value in dataset
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
