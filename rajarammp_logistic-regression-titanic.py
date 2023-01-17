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
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

train.head()
#Filling the missing values in Age

train['Age'].fillna(round(train["Age"].mean()),inplace=True)



#Converting the Sex column to numeric

assign = {'male':1, 'female':0}

train["Sex"] = train["Sex"].map(assign)



#Converting the Cabin column to numeric based on whether they had or not as Cabin indicator

train["Cabin_ind"] = np.where(train["Cabin"].isnull(),0,1)
#Checking the importance of Pclass, SibSp, Parch and Embarked

for i, col in enumerate(['Pclass', 'SibSp', 'Parch', 'Embarked']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=train, kind='point', aspect=1.5)
#Converting the Embarked column to numeric also Pclass to individual column since it is ordinal

pclass = pd.get_dummies(train['Pclass'])

embark = pd.get_dummies(train['Embarked'])

train = pd.concat([train, pclass, embark], axis=1)



#Concating Sibsp and Parch as Family_count 

train['family_cnt'] = train['SibSp'] + train['Parch']
#Checking the importance of Cabin_ind column

sns.catplot(x='Cabin_ind', y='Survived', data=train, kind='point')
#Checking the importance of Fare column

sns.catplot(x='Survived', y='Fare', data=train, kind='point')
train.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked","Pclass"],axis=1,inplace=True)

train.head()
#Splitting the training data to x_train and y_train

x_train = train.drop(['Survived'], axis=1)

y_train = train['Survived']
x_test = pd.read_csv('../input/titanic/test.csv')

x_test.head()
#Filling the missing values in Age and Fare

x_test['Age'].fillna(round(x_test["Age"].mean()),inplace=True)

x_test['Fare'].fillna(x_test["Fare"].mean(),inplace=True)



#Converting the Sex column to numeric

assign = {'male':1, 'female':0}

x_test["Sex"] = x_test["Sex"].map(assign)



#Converting the Cabin column to numeric based on whether they had or not as Cabin indicator

x_test["Cabin_ind"] = np.where(x_test["Cabin"].isnull(),0,1)



#Converting the Embarked column to numeric also Pclass to individual column since it is ordinal

pclass = pd.get_dummies(x_test['Pclass'])

embark = pd.get_dummies(x_test['Embarked'])

x_test = pd.concat([x_test, pclass, embark], axis=1)



#Concating Sibsp and Parch as Family_count 

x_test['family_cnt'] = x_test['SibSp'] + x_test['Parch']
x_test.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked","Pclass"],axis=1,inplace=True)

x_test.head()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(y_pred)
a = pd.DataFrame(np.array(y_pred), index=None)



pass_id = pd.DataFrame(np.arange(892, 1310, 1), index=None)



pred = pd.concat([pass_id, a], axis=1)

pred.columns = ['PassengerId', 'Survived']

pred
pred.to_csv('Prediction.csv', index=None)