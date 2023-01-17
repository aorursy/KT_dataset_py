

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
sns.set_style('whitegrid')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',data=train,hue='Pclass')
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
sns.countplot(x='SibSp', data=train)# majority with no spouse or children
train['Fare'].plot.hist(bins=40,figsize=(10,4))
import cufflinks as cf

cf.go_offline()
#train['Fare'].iplot(kind='hist',bins=40)
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=train)

# class 3 comprises mostly of young people than in first class
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train = pd.concat([train,sex,embark],axis=1)

train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head()

train.drop('PassengerId',axis=1,inplace=True)
train.head()
#pd.get_dummies(train['Pclass'])
X=train.drop('Survived',axis=1)

y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
predictions = log.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
train.info()
accuracy = (148+68)/889

accuracy

# 24% accuracy- very low