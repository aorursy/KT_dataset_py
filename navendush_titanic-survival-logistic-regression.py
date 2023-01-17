# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.isnull()
train_data.isnull()
train_data.info()
train_data.describe()
test_data.info()
test_data.describe()
sns.heatmap(train_data.isnull(),cbar=False,yticklabels=False)
sns.heatmap(test_data.isnull(),cbar=False,yticklabels=False)
sns.pairplot(train_data)
sns.set_style('whitegrid')

sns.countplot(x = 'Survived',data=train_data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train_data)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train_data)
sns.distplot(train_data['Age'].dropna(),kde=False,color='darkred',bins=30)
sns.countplot(x='SibSp',data=train_data)
train_data['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf

cf.go_offline()

train_data['Fare'].iplot(kind='hist',bins=30,color='green')
train_data['Age'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train_data)
train_data.groupby('Pclass').mean()
def add_age(columns):

    age = columns[0]

    Pclass = columns[1]

    

    if pd.isnull(age):

        if Pclass == 1:

            return 38.23

        elif Pclass == 2:

            return 29.87

        else:

            return 25.14

        

    else:

        return age

        
train_data['Age'] = train_data[['Age','Pclass']].apply(add_age,axis=1)

test_data['Age'] = test_data[['Age','Pclass']].apply(add_age,axis=1)
def add_fare(columns):

    fare = columns[0]

    Pclass = columns[1]

    

    if pd.isnull(fare):

        if Pclass == 1:

            return 84.15

        elif Pclass == 2:

            return 20.66

        else:

            return 13.67

        

    else:

        return fare
train_data['Fare'] = train_data[['Fare','Pclass']].apply(add_fare,axis=1)

test_data['Fare'] = test_data[['Fare','Pclass']].apply(add_fare,axis=1)
sns.heatmap(train_data.isnull(),cbar=False,yticklabels=False)
sns.heatmap(test_data.isnull(),cbar=False,yticklabels=False)
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train_data.isnull(),cbar=False,yticklabels=False)
sns.heatmap(test_data.isnull(),cbar=False,yticklabels=False)
sex_train = pd.get_dummies(train_data['Sex'],drop_first=True)

embark_train = pd.get_dummies(train_data['Embarked'],drop_first=True)

pclass_train = pd.get_dummies(train_data['Pclass'],drop_first=True)

sex_test = pd.get_dummies(test_data['Sex'],drop_first=True)

embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True)

pclass_test = pd.get_dummies(test_data['Pclass'],drop_first=True)
train_data = pd.concat([train_data,sex_train,embark_train,pclass_train],axis=1)

test_data = pd.concat([test_data,sex_test,embark_test,pclass_test],axis=1)
train_data.head()
train_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train_data.head()
test_data.head()
test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test_data.head()
train_data.drop(['PassengerId'],axis=1,inplace=True)

train_data.head()
train_data.columns
X_train = train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S',2, 3]]

y_train = train_data['Survived']

X_test = test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S',2, 3]]
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_logistic.csv', index=False)

print("Your submission was successfully saved!")