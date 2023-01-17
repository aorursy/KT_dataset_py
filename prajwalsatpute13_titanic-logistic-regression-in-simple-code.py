# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
# Let's check head,info & describe for train as we'll be fixing data & doing EDA for train_df first



train_df.head()
train_df.info()
train_df.describe()
train_df.isnull().sum()
sns.heatmap(train_df.isnull(),cmap='viridis',yticklabels=False)
sns.countplot(x='Survived',data=train_df,hue='Sex',palette='rainbow')
train_df[train_df['Survived']==1]['Survived'].sum()
sns.countplot(x='Survived',data=train_df,hue='Pclass',palette='plasma')
train_df['Age'].hist(bins=30,color='red',alpha=0.6)
train_df['Fare'].hist(bins=30,color='blue',alpha=0.6)
plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Age',data=train_df,palette='prism')
plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Fare',data=train_df,palette='prism_r')
train_df.isnull().sum()
class_1 = train_df[train_df['Pclass']==1]['Age'].mean()

class_2 = train_df[train_df['Pclass']==2]['Age'].mean()

class_3 = train_df[train_df['Pclass']==3]['Age'].mean()
train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==1), 'Age']=class_1

train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==2), 'Age']=class_2

train_df.loc[(train_df['Age'].isnull()) & (train_df['Pclass']==3), 'Age']=class_3
train_df.isnull().sum()
train_df.drop('Cabin',axis=1,inplace=True)
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()]
train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df.isnull().sum()
test_df.head()
test_df.info()
test_df.describe()
test_df.isnull().sum()
plt.figure(figsize=(10,6))

sns.heatmap(test_df.isnull(),cmap='GnBu_r',cbar=True,yticklabels=False)
test_df['Age'].hist(bins=30,alpha=0.6,color='orange')
test_df['Fare'].hist(bins=30,alpha=0.6,color='green')
plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Age',data=test_df,palette='nipy_spectral')
plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Fare',data=test_df,palette='twilight')
test_df.isnull().sum()
class_1_test = test_df[test_df['Pclass']==1]['Age'].mean()

class_2_test = test_df[test_df['Pclass']==2]['Age'].mean()

class_3_test = test_df[test_df['Pclass']==3]['Age'].mean()
test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==1),'Age'] = class_1_test

test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==2),'Age'] = class_2_test

test_df.loc[(test_df['Age'].isnull()) & (test_df['Pclass']==3),'Age'] = class_3_test
test_df.isnull().sum()
test_df[test_df['Fare'].isnull()]
meanFare_test = test_df.groupby('Pclass').mean()['Fare']

meanFare_test
test_df['Fare'] = test_df['Fare'].fillna(meanFare_test[3])
test_df.isnull().sum()
test_df.drop('Cabin',axis=1,inplace=True)
test_df.isnull().sum()
Sex = pd.get_dummies(train_df['Sex'],drop_first=True)

Embark = pd.get_dummies(train_df['Embarked'],drop_first=True)
train_df.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace=True)
train_df = pd.concat([train_df,Sex,Embark],axis=1)
train_df.head()
Sex = pd.get_dummies(test_df['Sex'],drop_first=True)

Embark = pd.get_dummies(test_df['Embarked'],drop_first=True)
test_df = pd.concat([test_df,Sex,Embark],axis=1)
test_df.drop(['Sex','Name','Ticket','Embarked',],axis=1,inplace=True)
test_df.head()
X_train = train_df.drop(['Survived','PassengerId'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)



X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': predictions})



submission.to_csv('Submission.csv', index = False)

submission.head()