import pandas as pd

import numpy as np

import sklearn

import re

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
#reading the datasets

train = pd.read_csv('../input/train.csv',skipinitialspace=True)

test = pd.read_csv('../input/test.csv',skipinitialspace=True)

full_data = [train,test]

PassengerId = test['PassengerId']

train.head(5)

test.tail(5)
train.head(10)
#exploratory statistics

train.describe()
#exploratory statistics

test.describe()
train.info()
train.isnull().any()
test.info()
test.isnull().any()
#featuring engineering. Adding new columns

for dataset in full_data:

    dataset['hasCabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    dataset['isAlone'] = 0

    dataset.loc[dataset['SibSp']+dataset['Parch'] == 0 , 'isAlone'] = 1

train.head(5)
#filling NAs

for dataset in full_data:

    null_age = dataset['Age'].isnull().sum()

    dataset['Age'][np.isnan(dataset['Age'])] = null_age

    dataset['Age'] = dataset['Age'].astype(int)

for dataset in full_data:

    null_embarked = dataset['Embarked'].isnull().sum()

    dataset['Embarked'][pd.isnull(dataset['Embarked'])] = null_embarked

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

for dataset in full_data:

    null_sex = dataset['Sex'].isnull().sum()

    dataset['Sex'][pd.isnull(dataset['Sex'])] = null_sex
train.head(5)
#dropping unnecessary columns

for dataset in full_data:

    del dataset['PassengerId']

    del dataset['Name']

    del dataset['Ticket']

    del dataset['Cabin']
train.head(5)
#mapping sex

for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].apply(lambda x: 0 if x == 'female' else 1)
#mapping embarked

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: 0 if x=='S' else 1 if x =='C' else 2)

train.head(5)
#correlation plots

colormap = plt.cm.viridis

plt.figure(figsize=(10,10))

plt.title("Correlation Matrix")

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
#logistic regression model

from sklearn.linear_model import LogisticRegression

predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','hasCabin','isAlone']
model = LogisticRegression(random_state = 1)

model.fit(train[predictors],train['Survived'])

predictions = model.predict(test[predictors])
model.coef_
submission = pd.DataFrame(

    {'PassengerId':PassengerId,

     'Survived': predictions

    })

submission.to_csv("kaggle.csv", index=False)
submission.head(5)