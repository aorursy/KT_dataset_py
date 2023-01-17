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
#Importing libraries

import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt
#Retrieving training and test data

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

both = [train_df, test_df]
#Columns

print('train columns \n')

print(train_df.columns.values)

print('\ntest columns \n')

print(test_df.columns.values)
train_df.info()
train_df.describe()
test_df.describe()
train_df.describe(include=['O'])
#train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

import seaborn as sns
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
sb.barplot(x='Pclass', y='Survived', data=train_df, color='r')
sb.barplot(x='Sex', y='Survived', data=train_df, color='r')
sb.barplot(x='Parch', y='Survived', data=train_df, color='r')
sb.barplot(x='Survived', y='Fare', data=train_df, color='r')
train_df = train_df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])

test_Ids = test_df['PassengerId']

test_df = test_df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'])
train_df.head()
for cols in train_df:

    print("col : {} -- {}= {}".format(type(train_df[cols][0]),cols,train_df[cols].isnull().sum()))
for cols in test_df:

    print("col : {} -- {}= {}".format(type(test_df[cols][0]),cols,test_df[cols].isnull().sum()))
train_df = train_df.fillna(train_df['Age'].mean())

train_df = train_df.fillna(train_df['Embarked'].mode())



test_df = test_df.fillna(test_df['Age'].mean())

test_df = test_df.fillna(test_df['Fare'].mean())
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

encoder = LabelEncoder()

train_df['Sex'] = encoder.fit_transform(train_df['Sex'].astype(str))

train_df['Embarked'] = encoder.fit_transform(train_df['Embarked'].astype(str))



test_df['Sex'] = encoder.fit_transform(test_df['Sex'].astype(str))

test_df['Embarked'] = encoder.fit_transform(test_df['Embarked'].astype(str))



test_df.head()
#Getting training data using train test split

x_train, x_val, y_train, y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], 

                                                 test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
#Logistic Regression

lg = LogisticRegression()

lg.fit(x_train, y_train)

y_pred = lg.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Gaussian Naive Bayes

GNB = GaussianNB()

GNB.fit(x_train, y_train)

y_pred = GNB.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Random Forest

RF = RandomForestClassifier(criterion='entropy',n_estimators=200,max_depth=8,random_state=7,

                            class_weight='balanced')

RF.fit(x_train, y_train)

y_pred = RF.predict(x_val)

print(accuracy_score(y_pred, y_val))
RF = RandomForestClassifier(n_estimators=300)

RF.fit(x_train, y_train)

y_pred = RF.predict(x_val)

print(accuracy_score(y_pred, y_val))
RF250 = RandomForestClassifier(criterion='entropy',n_estimators=250,max_depth=9,random_state=7,

                            class_weight='balanced')

RF250.fit(x_train, y_train)

y_pred = RF250.predict(x_val)

print(accuracy_score(y_pred, y_val))
RF300 = RandomForestClassifier(criterion='entropy',n_estimators=300,max_depth=9,random_state=7,

                            class_weight='balanced')

RF300.fit(x_train, y_train)

y_pred = RF300.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Support Vector Machines

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

print(accuracy_score(y_pred, y_val))
#KNearest Neighbour

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Perceptron

pn = Perceptron()

pn.fit(x_train, y_train)

y_pred = pn.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Linear SVC

linSVC = LinearSVC()

linSVC.fit(x_train, y_train)

y_pred = linSVC.predict(x_val)

print(accuracy_score(y_pred, y_val))
tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)

y_pred = tree.predict(x_val)

print(accuracy_score(y_pred, y_val))
#Random Forest 250 estimators, most accurate model



model = RandomForestClassifier(criterion='entropy',n_estimators=200,max_depth=8,random_state=7,

                            class_weight='balanced')

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

print(accuracy_score(y_pred, y_val))



#retrieving passenger Ids

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test_passenger_Id = test['PassengerId']
predictions = model.predict(test_df)

output = pd.DataFrame({'PassengerId': test_passenger_Id, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")