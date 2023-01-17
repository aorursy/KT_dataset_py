# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import datasets, linear_model, metrics 

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import metrics as m



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

valid = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.info()

#test.info()
train.head()

#test.head()
train.isnull().sum()

#test.isnull().sum()
total = train.isna().sum().sort_values(ascending=False)

percent = ((train.isna().sum()/train.isna().count())*100).sort_values(ascending=False)



missing_data = pd.concat([total,percent],axis=1,keys=['total','percent'])

missing_data
train['Survived'].groupby(train['Age'].isnull()).mean()
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Survived'].groupby(train['Cabin'].isnull()).mean()
train['Cabin_ind'] = np.where(train['Cabin'].isnull(),0,1)
total = test.isna().sum().sort_values(ascending=False)

percent = ((test.isna().sum()/test.isna().count())*100).sort_values(ascending=False)



missing_data = pd.concat([total,percent],axis=1,keys=['total','percent'])

missing_data
test['Cabin_ind'] = np.where(test['Cabin'].isnull(),0,1)
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
for i,col in enumerate(['Cabin_ind','Sex','Embarked']):

    plt.figure(i)

    sns.catplot(x=col,y='Survived',data=train,kind='point',aspect=2)

    
for i,col in enumerate(['Pclass','SibSp','Parch']):

    plt.figure(i)

    sns.catplot(x=col,y='Survived',data=train,kind='point',aspect=2,)
train['Family_count'] = train['SibSp']+train['Parch']
test['Family_count'] = test['SibSp']+test['Parch']
train.drop(['SibSp','Parch'],axis=1,inplace=True)



test.drop(['SibSp','Parch'],axis=1,inplace=True)

train.head()

#test.head()
train.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
print(train.head())

test.head()
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.head())

test.head()
x_features = train.drop('Survived',axis=1)

y_targets = train['Survived']
test_target = valid['Survived']

test_data = test.drop('PassengerId',axis=1)
x_features.head()
test_data.head()
tree_model = DecisionTreeClassifier(max_depth=5,random_state=0)

tree_model.fit(x_features,y_targets)
tree_prediction = tree_model.predict(test_data)

acc = accuracy_score(test_target,tree_prediction)

acc
confusion_matrix(tree_prediction,test_target)
m.precision_score(tree_prediction,test_target)
m.recall_score(tree_prediction,test_target)
forest_model = RandomForestClassifier(n_estimators=300,max_depth=3,random_state=0)

forest_model.fit(x_features,y_targets)
forest_prediction = forest_model.predict(test_data)

acc = accuracy_score(test_target,forest_prediction)

acc
confusion_matrix(forest_prediction,test_target)
m.precision_score(forest_prediction,test_target)
m.recall_score(forest_prediction,test_target)
y_pred = forest_model.predict(test_data)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
output