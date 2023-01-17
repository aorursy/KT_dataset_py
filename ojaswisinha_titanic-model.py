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
#Libraries imported

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.shape
data.columns
data.drop('PassengerId', axis =1, inplace = True)
data.info()
data['Survived'].value_counts()
data['Pclass'].value_counts()
grp = data.groupby(['Pclass', 'Survived']).Sex.value_counts()

grp
grp = data.groupby(['Pclass', 'Survived']).Sex.value_counts().plot(kind = 'bar', color = 'green',)

data.isnull().any()
data['Age'].mean()
data['Age'].fillna(29.6, inplace = True)

data.isnull().any()
data.drop('Cabin', inplace = True, axis = 1)

data.isnull().any()
data['Embarked'].isnull().value_counts()
data['Embarked'].value_counts()
data['Embarked'].fillna('S', inplace = True)
fig = plt.figure(1)

ax = fig.add_subplot(111)

data.groupby('Survived').Sex.value_counts().plot(kind = 'bar', title = 'Survival count based on Sex')

plt.xlabel("Survived-Sex")

xticklabels = ['Not-Sur-M', 'Not-Sur-F', 'Sur-F', 'Sur-M']

ax.set_xticklabels(xticklabels, rotation = 45)

plt.ylabel('Counts')  
fig = plt.figure(1)

ax = fig.add_subplot(111)

data.groupby('Survived').Pclass.value_counts().plot(kind = 'bar', title = 'Survival count based on Sex')

plt.xlabel("Survived-Sex")

#xticklabels = ['Not-Sur-M', 'Not-Sur-F', 'Sur-F', 'Sur-M']

#ax.set_xticklabels(xticklabels, rotation = 45)

plt.ylabel('Counts') 
sns.catplot( x= 'Pclass', hue = 'Sex',col = 'Survived', data = data, kind = 'count')
data.groupby('Survived').Fare.value_counts()

sns.catplot(x = 'Fare', col = 'Survived', data =data , kind = 'strip')
sns.catplot(x = 'Age', col = 'Survived', row = 'Sex', data =data , kind = 'violin')
data.groupby('Sex').Age.max()
sns.catplot(x = 'Embarked', col = 'Survived', data =data , kind = 'count')
data.groupby('Embarked').Survived.value_counts()
sns.catplot(x = 'SibSp', col = 'Pclass', row = 'Survived', data =data , kind = 'count')
sns.catplot(x = 'Parch', col = 'Pclass', row = 'Survived', data =data , kind = 'count')
sex = pd.get_dummies(data['Sex'], drop_first = True)

embark = pd.get_dummies(data['Embarked'], drop_first = True)

data.drop(['Sex', 'Embarked','Name', 'Ticket'], axis = 1, inplace = True)

data = pd.concat([data, sex, embark], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis = 1),data['Survived'], test_size = 0.30, random_state = 101)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report, roc_curve, auc

result = classification_report(y_test, pred)

print(result)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
#for test data

test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)

test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data.drop(['Sex','Embarked','Name','Ticket', 'Cabin'],axis=1,inplace=True)

test_data = pd.concat([test_data,test_sex,test_embark],axis=1)

#fill null value of fare column with 0

test_data.Fare.fillna(0 ,inplace = True)

test_data.head()
#test_data.drop('Cabin', axis = 1, inplace = True)

test_data.head()

test_data['Age'].mean()

test_data['Age'].fillna(30, inplace = True)

test_data.isnull().any()
id = test_data['PassengerId']

predictions = model.predict(test_data.drop('PassengerId', axis=1))

result = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions })

result.head()