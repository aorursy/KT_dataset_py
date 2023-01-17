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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Load the data

train= pd.read_csv('../input/train.csv', header=0)

test= pd.read_csv('../input/test.csv', header=0)

train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf

cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

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
# now apply this function to both train and test datasets

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
# Now let's check that heat map again! for both the datasets

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
test.info()
test.isna().sum()
test['Fare'].fillna(method='bfill',inplace=True)
#train dataset

sex_train = pd.get_dummies(train['Sex'],drop_first=True)

embark_train = pd.get_dummies(train['Embarked'],drop_first=True)

#test dataset

sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test = pd.get_dummies(test['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex_train,embark_train],axis=1)

test = pd.concat([test,sex_test,embark_test],axis=1)
train.head()
X_train=train.drop('Survived',axis=1)

y_train=train['Survived']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='warn')

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(test)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("titanic_kaggle_submission.csv", index=False)
print(submission)