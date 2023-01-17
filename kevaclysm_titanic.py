# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)
df1 = pd.read_csv('../input/test.csv', header=0)


import pylab as P
df['Age'].hist()
P.show()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df1['Gender'] = df1['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df.head(10)
median_ages = np.zeros((2,3))
median_ages1 = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
        median_ages1[i,j] = df1[(df1['Gender'] == i) & \
                              (df1['Pclass'] == j+1)]['Age'].dropna().median()
 



df1['AgeFill'] = df1['Age']

df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
        df1.loc[ (df1.Age.isnull()) & (df1.Gender == i) & (df1.Pclass == j+1),\
                'AgeFill'] = median_ages1[i,j]

df['FamilySize'] = df['SibSp'] + df['Parch']
df1['FamilySize'] = df1['SibSp'] + df1['Parch']


df['Age*Class'] = df.AgeFill * df.Pclass
df1['Age*Class'] = df1.AgeFill * df1.Pclass

df['EmbarkedCode'] = df['Embarked'].dropna().map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df1['EmbarkedCode'] = df1['Embarked'].dropna().map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df1 = df1.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

df = df.drop(['Age'], axis=1)
df1 = df1.drop(['Age'], axis=1)

df = df.dropna()
df1 = df1.dropna()

train_data = df.values
test_data = df1.values

from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

output
