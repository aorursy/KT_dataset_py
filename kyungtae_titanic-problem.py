%matplotlib inline
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
#train.to_csv('copy_of_the_training_data.csv', index=False)
X_train = train[['Sex', 'Pclass', 'Fare', 'Parch', 'Age', 'Cabin', 'Embarked']].as_matrix()
X_train
train[train['Age'].isnull()][['Sex', 'Pclass', 'Age']]
for i in range(1,4):
    print (i, len(train[ (train['Sex'] == 'male') & (train['Pclass'] == i) ]))
import pylab as P

train['Age'].hist()
P.show()
train['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()
train['Gender'] = 4
train['Gender'] = train['Sex'].map( lambda x: x[0].upper() )
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                              (train['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
train['AgeFill'] = train['Age']
train[ train['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
train[ train['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
train.describe()
train['FamilySize'] = train['SibSp'] + train['Parch']

train['Age*Class'] = train.AgeFill * train.Pclass
train['Age*Class']
train.dtypes[train.dtypes.map(lambda x: x=='object')]
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
train = train.drop(['Age'], axis=1)
train = train.dropna()
train_data = train.values
train_data

import pprint
pprint.pprint(train, width=400)