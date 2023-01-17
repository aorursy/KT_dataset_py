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
#importing pakages

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
#importing dataset

dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

dataset.head()
Dataset = "titanic"
dataset.shape
#check for missing values

sns.heatmap(dataset.isnull())
#to check number of people that survived the disaster and number of people who did not survive.

sns.countplot(x='Survived',data=dataset)
#number of people that survived based on their age.

sns.countplot(x='Survived',data=dataset,hue="Sex")
#number of people that survived based on their Pclass.

sns.countplot(x='Survived',data=dataset,hue="Pclass")
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=dataset,palette='winter')
#missing value imputation on age and Pclass

def impute_age(cols):

  Age = cols[0]

  Pclass = cols[1]

  

  if pd.isnull(Age):

    if Pclass == 1:

      return 37

    elif Pclass ==2:

      return 29

    else:

      return 24

  else:

    return Age
dataset['Age']= dataset[['Age','Pclass']].apply(impute_age,axis=1)
#dropping independent variable cabin

dataset.drop('Cabin',axis =1,inplace=True)
dataset.groupby('Embarked').size()
#imputing common value for embarked

common_value = 'S'

dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

dataset.info()
dataset.info()
dataset.head()
sex = pd.get_dummies(dataset['Sex'],drop_first=True)

embark = pd.get_dummies(dataset['Embarked'],drop_first=True)
dataset.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset = pd.concat([dataset,sex,embark],axis=1)
#Creating training data set

X_train = dataset.drop(['Survived'],axis =1 )

y_train = dataset['Survived']
#USING RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)

y_pred_rf=rf.predict(X_train)
from sklearn import metrics

print('Accuracy',metrics.accuracy_score(y_train,y_pred_rf))
#importing training data

data = pd.read_csv('/kaggle/input/titanic/test.csv')

data.shape
passengerID = data['PassengerId']
data.info()
sns.heatmap(data.isnull())
#missing value imputation on age and Pclass

def impute_age(cols):

  Age = cols[0]

  Pclass = cols[1]

  

  if pd.isnull(Age):

    if Pclass == 1:

      return 37

    elif Pclass ==2:

      return 29

    else:

      return 24

  else:

    return Age
data['Age']= data[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(data.isnull())
#dropping cabin from dataset

data.drop('Cabin',axis =1,inplace=True)
data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)

data.fillna(method='ffill',inplace=True)
sex = pd.get_dummies(data['Sex'],drop_first=True)

embark = pd.get_dummies(data['Embarked'],drop_first=True)
data.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
data = pd.concat([data,sex,embark],axis=1)
data.head()
#predicting the people who survived on test data

y_pred_test = rf.predict(data)
y_pred_test
df = pd.DataFrame({'PassengerID':passengerID, 'Survived':y_pred_test})

df