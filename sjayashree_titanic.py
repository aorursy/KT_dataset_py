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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.shape
train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


test_data.head(2)
test_data.shape
#Missing data analysis

import seaborn as sns



sns.heatmap(train_data.isnull(),yticklabels=False)

#Dropping cabin column as there is so much missing data

train_data.drop('Cabin',inplace=True,axis=1)

test_data.drop('Cabin',inplace=True,axis=1)
train_data.head(2)
test_data.head(2)
sns.countplot(x='Survived',data=train_data,hue='Sex')
train_data.info()
test_data.info()
train_data.describe()
#Lets see the categorical features

train_data.describe(include=['O'])
train_data.drop(['Name','Ticket'],inplace=True,axis=1)
train_data.head(2)
test_data.drop(['Name','Ticket'],inplace=True,axis=1)
test_data.head(2)
gender = pd.get_dummies(train_data['Sex'],drop_first=True)
embark = pd.get_dummies(train_data['Embarked'],drop_first=True)
train_data.drop(['Sex','Embarked'],axis=1,inplace=True)
train_data.head(2)
train_data=pd.concat([train_data,gender,embark],axis=1)
train_data.head(2)
gender_test = pd.get_dummies(test_data['Sex'],drop_first=True)

embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data.drop(['Sex','Embarked'],axis=1,inplace=True)

test_data=pd.concat([test_data,gender_test,embark_test],axis=1)
test_data.head()
train_data.describe()
new_age_test = train_data['Age'].fillna(29)
train_data['Age'] = new_age_test
test_data['Age'].fillna(30,inplace=True)
train_data.info()

print('-'*40)

test_data.info()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_data.drop('Survived',axis=1),train_data['Survived'])
test_data.describe()
test_data.info()
test_data['Fare'].fillna(35,inplace=True)
predictions = model.predict(test_data)
output= pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})



output.to_csv('output.csv',index=False)