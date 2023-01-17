import pandas as pd

import numpy as np





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



combine = [train, test]
train.describe()
train.describe(include=['O'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
for data in combine:

    data['Title'] = data.Name.str.extract(' ([a-zA-Z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
train.corr()
train.drop(['Ticket', 'Cabin'], axis=1)

test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]
for ds in combine:

    ds['Title'] = ds['Title'].replace(['Lady','Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    ds['Title'] = ds['Title'].replace('Mlle', 'Miss')

    ds['Title'] = ds['Title'].replace('Ms', 'Miss')

    ds['Title'] = ds['Title'].replace('Mme', 'Mrs')



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)



train.head()
rando = np.zeros((2,3))



for d in combine:

    for i in range(0,2):

        for j in range(0,3):            

            b = d[(d['Sex'] == i) & (d['Pclass'] == j+1)]['Age'].dropna()

            

            guess = b.median()

            rando[i,j] = int(guess/.5 + .5) * .5



    for i in range(0,2):

        for j in range(0,3):

            d.loc[(d.Age.isnull()) & (d.Sex == i) & (d.Pclass == j+1), 'Age'] = rando[i,j]

    

    d['Age'] = d.Age.astype(int)

             

train.head()





train = train.drop(['Name', 'PassengerId'], axis=1)

test  = test.drop(['Name'], axis =1)

combine = [train, test]


