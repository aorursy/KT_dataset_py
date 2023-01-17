import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
train.info()
train.isnull().tail()
type(train.Age.isnull())
train[train['Age'].isnull()][['Sex', 'Pclass', 'Age']]
for i in range(1,4):
    print(i, len(train[(train['Sex'] == 'male') & (train['Pclass'] == i)]))
type((train['Sex'] == 'male') & (train['Pclass'] == i))
import pylab as p
train['Age'].hist()
train['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
train.shape
train['Gender'] = 0
train.tail(n=4)
train['Gender'] = train['Sex'].map(lambda x: x[0].upper())
train.Gender = train.Sex
train.head()
train.Gender = train['Sex'].map(lambda x: x[0].upper())
train.Gender.head()
train['Onboard'] = 'Q'
train.dtypes
train.Onboard = train.Embarked.map(lambda x: x)
train.head()
train.Gender = train.Sex.map({'female': 0, 'male': 1}).astype(int)
train.head()
median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0,3):
        median_ages[i,j] = train[(train['Gender'] == i) & (train['Pclass'] == (j + 1))]['Age'].dropna().median()
median_ages
train.columns
train.info()
train['AgeFill'] = train.Age
train.info()
train[train['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(n=10)
for i in range(0,2):
    for j in range(0,3):
        train.loc[(train['Gender'] == i) & (train.Pclass == (j+1)), 'AgeFill'] = median_ages[i,j]
train[train['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(n=10)
train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
train.AgeIsNull
train['FamilySize'] = train['SibSp'] + train['Parch']
train['Age*Pclass'] = train['AgeFill'] * train.Pclass
train.dtypes
train.info()
type(train.dtypes)
train.dtypes[train.dtypes == 'object']
train = train.drop(['Name', 'Sex', 'Tickets', 'Cabin', 'Embarked', 'Onboard'])
tr


