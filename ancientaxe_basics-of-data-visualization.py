import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



data = pd.read_csv('../input/train.csv')

data.sample(n=10)
data.Sex = pd.Categorical(data.Sex)

data['Gender'] = data.Sex.cat.codes

#data = data.drop(['Sex'], axis=1)

#data.sample(n=5)
dfsex = data[['PassengerId', "Sex", "Survived"]].groupby(['Sex', 'Survived']).count()/data[['PassengerId', "Sex"]].groupby(['Sex']).count()

dfsex
x = data['Survived']

plt.ylabel('Count')

plt.xticks([0.25,0.75],['Deceased','Survived'])

plt.hist(x, bins=2)
sns.countplot(x='Sex',data=data)

sns.despine()
sns.barplot(x='Sex', y='Survived',data=data)

sns.despine()
sns.barplot(x='Embarked', y='Survived',data=data)

sns.despine()
sns.distplot(data['Fare'])

sns.despine()