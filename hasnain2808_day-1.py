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
import os
import pandas 
i=5

print(i)
i=6 

print(i)
# Name,Rollno

# Moha,45,

# keith,85
import pandas as pd
pd.read_csv('/kaggle/input/titanic/train.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head(10)
train.describe()
train.dtypes
train
print(train['Survived'])
train.iloc[0:5,0:2]
train.loc[0:5,'Survived']
help(train)
train
train = train.drop(['PassengerId', 'Name'], axis =1 )
train

print('Survived: ' + str( len(train['Survived']) - train['Survived'].count()))

print('Pclass: ' + str( len(train['Pclass']) - train['Pclass'].count()))

print('Sex: ' + str( len(train['Sex']) - train['Sex'].count()))

print('Age: ' + str( len(train['Age']) - train['Age'].count()))

print('SibSp: ' + str( len(train['SibSp']) - train['SibSp'].count()))

print('Parch: ' + str( len(train['Parch']) - train['Parch'].count()))

print('Ticket: ' + str( len(train['Ticket']) - train['Ticket'].count()))

print('Fare: ' + str( len(train['Fare']) - train['Fare'].count()))

print('Cabin: ' + str( len(train['Cabin']) - train['Cabin'].count()))

print('Embarked: ' + str( len(train['Embarked']) - train['Embarked'].count()))
train = train.drop(['Cabin'],axis =1 )
train['Age'].mean()
train['Age'] = train['Age'].fillna(train['Age'].mean())
print('Age: ' + str( len(train['Age']) - train['Age'].count()))

train['Embarked'].count()
train['Embarked'] = train['Embarked'].fillna('S')
str(len(train['Age']) - train['Age'].count())
print('Age: ' + str( len(train['Age']) - train['Age'].count()))

train['Embarked'].value_counts()
train.dtypes
set(train['Sex'])
sex_dictionary = {'female':0, 'male':1}
train['Sex'] = train['Sex'].map(sex_dictionary)
train['Sex']
train.dtypes
set(train['Embarked'])
Embarked_dictionary = {'C' : 0, 'Q' : 1, 'S' : 2}
train['Embarked'] = train['Embarked'].map(Embarked_dictionary)
train
[[1,2,3,4,6,7,3,2]]
train = train.drop('Ticket', axis = 1)
train.columns
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

train['Age'].hist()
category = pd.cut(train.Age,bins=[0,2,20,60,80],labels=[0,1,2,3])

train['Age_category'] = category

print(category, train.Age)
train.columns
max(train.Age)
train.columns
import matplotlib.pyplot as plt

%matplotlib inline


plt.hist([train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], stacked=False, bins = 10, label = ['Survived','Non-survived'])

plt.xlabel('ticket Fare')

plt.ylabel('Survived and non survived')

plt.legend()

plt.show()
plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=False, bins = 8, label = ['Survived','Non-survived'])

plt.xlabel('Age')

plt.ylabel('Survived and non survived')

plt.legend()

plt.show()
import seaborn as sns
sns.countplot(x='Survived', data=train);

sns.countplot(x='Sex', data=train);

train = train.sample(frac=1).reset_index(drop=True)

train
X = train[['Sex','Age_category', 'Fare', 'Embarked', 'FamilySize','Pclass']]
X
Y = train['Survived']
Y
X[:600]
from sklearn.linear_model import LogisticRegression

logisticregression = LogisticRegression()

logisticregression.fit(X[:600],Y[:600])
help(LogisticRegression)
from sklearn.metrics import accuracy_score, classification_report
pred_logreg = logisticregression.predict(X[:600])

print(classification_report(Y[:600], pred_logreg))

print(accuracy_score(Y[:600], pred_logreg))
pred_logreg = logisticregression.predict(X[600:])

print(classification_report(Y[600:], pred_logreg))

print(accuracy_score(Y[600:], pred_logreg))
from sklearn.tree import DecisionTreeClassifier

DecisionT = DecisionTreeClassifier(max_leaf_nodes = 20)

DecisionT.fit(X[:600],Y[:600])
pred_dec_tree = DecisionT.predict(X[:600])

print(classification_report(Y[:600:], pred_dec_tree))

print(accuracy_score(Y[:600], pred_dec_tree))
pred_dec_tree = DecisionT.predict(X[600:])

print(classification_report(Y[600:], pred_dec_tree))

print(accuracy_score(Y[600:], pred_dec_tree))
import pickle
from joblib import dump, load

dump(DecisionT, 'DecisionT.joblib') 
DecisionT = load('DecisionT.joblib') 

