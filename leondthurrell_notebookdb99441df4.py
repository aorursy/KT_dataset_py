import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



input_dir = '../input/'





training = pd.read_csv('../input/train.csv')

testing = pd.read_csv(input_dir + "test.csv")
training.head(10)
print('Total Rows: '+str(len(training)))

print(training.isnull().sum())
print(training['Cabin'].isnull().sum()/len(training)*100)
training = training.drop('Cabin', 1)

training = training.drop('Ticket', 1)

training['Embarked'].fillna('S', inplace=True)

training = training.drop('Name', 1)

training.head(5)
print(training['Age'].isnull().sum()/len(training)*100)
ax = training["Age"].hist(bins=15, color='teal', alpha=0.8)

ax.set(xlabel='Age', ylabel='Count')

plt.show()
training['Age'] = training['Age'].fillna(training['Age'].median())

training.head(10)
MF = ['male', 'female']

port = ['C','S','Q']

training ['Sex'] = training['Sex'].astype("category", categories=MF).cat.codes

training ['Embarked'] = training['Embarked'].astype("category", categories=port).cat.codes

training.head(5)
train2 = pd.get_dummies(training, columns=["Pclass"])

train3 = pd.get_dummies(train2, columns=["Embarked"])

train3.head(5)