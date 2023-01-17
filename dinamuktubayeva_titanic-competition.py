import pandas as pd

import numpy as np

import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt



import sklearn

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

total = train + test
print('Columns:')

print(total.columns.values)
train.head()
train.info()

print( )

test.info()
total.describe()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
age_bar = sns.FacetGrid(train, col='Survived')

age_bar.map(plt.hist, 'Age', bins=20)
fare_plot = sns.FacetGrid(train, hue="Survived",aspect=4)

fare_plot.map(sns.kdeplot,'Fare',shade= True)

fare_plot.set(xlim=(0, train['Fare'].max()))

fare_plot.add_legend()

 

plt.show()  
print('Missing in the train data: ')

display(train.isnull().sum())

print('Missing in the test data: ')

display(test.isnull().sum())
train["Age"] = train["Age"].fillna((train["Age"].median()))

test["Age"] = test["Age"].fillna((test["Age"].median()))

test['Fare'] = test['Fare'].fillna((test['Fare'].median()))
print('Missing in the train data: ')

display(train.isnull().sum())

print('Missing in the test data: ')

display(test.isnull().sum())
X_train = train[['Fare', 'Age']].values

Y_train = train['Survived'].values

X_test = test[['Fare', 'Age']].values
model = LogisticRegression()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

accuracy = round(model.score(X_train, Y_train) * 100, 2)

accuracy
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")