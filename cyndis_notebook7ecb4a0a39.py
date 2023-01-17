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
import pandas as pd





# For .read_csv, always use header=0 when you know row 0 is the header row

df = pd.read_csv('../input/train.csv', header=0)

type(df)
df2 = pd.read_csv('../input/test.csv', header=0)

df2.head()

test_data = df2.values
import pylab as P

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)

P.show()
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df.head()
df2['Gender'] = df2['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df2.head()
print (df.Embarked.unique())

df['EmbarkedInt'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} )

df.head()
df2['EmbarkedInt'] = df2['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} )

df2.head()
median_ages = np.zeros((2,3))

median_ages
for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

 

median_ages
df['AgeFill'] = df['Age']



df.head()
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)



df.describe()
df2['AgeIsNull'] = pd.isnull(df2.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
for i in range(0, 2):

    for j in range(0, 3):

        df2.loc[ (df2.Age.isnull()) & (df2.Gender == i) & (df2.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]



df2['FamilySize'] = df2['SibSp'] + df2['Parch']

df2['Age*Class'] = df2.AgeFill * df2.Pclass

df2 = df2.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 

df2 = df2.drop(['Age'], axis=1)
df2 = df2.drop(['PassengerId'], axis=1)
df = df.drop(['Age'], axis=1)
df = df.dropna()

df = df.drop(['PassengerId'], axis=1)

df.head()
train_data = df.values

train_data
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)



# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])



# Take the same decision trees and run it on the test data

output = forest.predict(test_data)