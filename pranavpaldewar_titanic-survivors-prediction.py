# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data = pd.read_csv('../input/titanic/train.csv')

test_data.head()

test_data.shape

test_data.corr()

test_data['Age'].fillna(test_data['Age'].mean())

test_data['Embarked'].fillna('Q')

test_data['Age'].isnull().sum()

test_data['Embarked'].isnull().sum()

test_data.info()
import seaborn as sns

from matplotlib import pyplot as plt

plt.title('no of survivors')

sns.countplot(x =test_data["Sex"])





plt.figure(num =1)

plt.title('no. of people embarking at different points')

sns.countplot(x = test_data['Embarked'])

plt.figure(num=2)

plt.title('no. of people in each class')

sns.countplot(x = test_data['Pclass'])

plt.figure(num =1)

plt.title('no. of survivors from differnt categories of embarked')

sns.countplot(x = test_data[test_data['Survived']==1]['Embarked'])

plt.figure(num=2)

plt.title('no. of survivors in each class')

sns.countplot(x = test_data[test_data['Survived']==1]['Pclass'])

features=['Pclass','Age','Parch','SibSp','Fare','Sex','Embarked']

test = pd.get_dummies(test_data[features])

test.Age =test['Age'].fillna(test_data.Age.mean())



test.info()
sns.distplot(test_data[test_data['Survived']==1]['Age'])

sns.distplot(test_data[test_data['Survived']==0]['Age'])
from sklearn.model_selection import train_test_split as ts

from sklearn.ensemble import RandomForestClassifier

x=test

y=test_data['Survived']

s_model = RandomForestClassifier(n_estimators=10)

train_x,val_x,train_y,val_y =ts(x,y,random_state=0)

s_model.fit(train_x,train_y)

pred=s_model.predict(val_x)
from sklearn.metrics import mean_absolute_error,accuracy_score

print(mean_absolute_error(val_y,pred))

print(s_model.score(train_x,train_y)*100)
t_data=pd.read_csv('../input/titanic/test.csv')

t_pred_data=t_data[features]

t_pred = pd.get_dummies(t_pred_data)

t_pred.Age=t_pred.Age.fillna(t_pred.Age.mean())

t_pred.Fare=t_pred.Fare.fillna(t_pred.Fare.mean())

t_pred = s_model.predict(t_pred)
submission= t_data['PassengerId']

submission_csv =pd.DataFrame({'PassengerId':submission,'Survived':t_pred})

submission_csv.to_csv('Submission_csv',index=False)