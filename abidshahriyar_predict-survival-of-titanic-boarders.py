import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import tree

%matplotlib inline
traindf = pd.read_csv(r'../input/titanic/train.csv')

null_values = traindf.isnull().sum()

print('Check for null Values :\n', null_values)

print('Check basic statistics: \n', traindf.describe(include='all'))

print('Check number of unique entries: \n', traindf.nunique())

print('Check for Data Types: \n', traindf.dtypes)
traindf = traindf.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'SibSp', 'Parch', 'Age'], axis=1)

traindf = traindf.dropna(axis=0, subset=['Embarked'])

traindf.Sex = traindf.Sex.map({'male': 0, 'female': 1})

traindf.Embarked = traindf.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

traindf
correlation_matrix = traindf.corr()

correlation_matrix['Survived'].sort_values(ascending=False)
featuresdf = traindf[list(traindf.columns[1:])]

targetdf = traindf['Survived']

clf = tree.DecisionTreeClassifier()

model = clf.fit(featuresdf, targetdf)

plt.title('Decision Tree')

tree.plot_tree(model, feature_names=list(traindf.columns[1:]), class_names=['Died', 'Survived'], filled=True, rounded=True,

              proportion=True, rotate=True)
titanictestdf = pd.read_csv(r'../input/titanic/test.csv')

testdf = titanictestdf.drop(['Cabin', 'Name', 'PassengerId', 'Ticket','SibSp', 'Parch', 'Age'], axis=1)

testdf = testdf.dropna(axis=0, subset=['Embarked'])

testdf.Sex = testdf.Sex.map({'male': 0, 'female': 1})

testdf.Embarked = testdf.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

testdf
testdf.isnull().sum()
testdf.loc[testdf[testdf['Fare'].isnull()].index.to_list()]
titanictestdf.loc[152]
sns.swarmplot(traindf.Pclass, traindf.Fare)

traindf.groupby('Pclass')['Fare'].agg(np.mean)
testdf.dropna().groupby('Pclass')['Fare'].agg(np.mean)
testdf.loc[152, 'Fare'] = np.mean([testdf.dropna().groupby('Pclass')['Fare'].agg(np.mean).loc[3],traindf.dropna().groupby('Pclass')['Fare'].agg(np.mean).loc[3]])

testdf.loc[152,'Fare'] 
model.predict(testdf)
survival_prediction = titanictestdf.copy()

survival_prediction['Survived'] = model.predict(testdf)

survival_prediction = survival_prediction[['PassengerId', 'Survived']].set_index('PassengerId')

survival_prediction
survival_prediction.to_csv('survival_prediction.csv')