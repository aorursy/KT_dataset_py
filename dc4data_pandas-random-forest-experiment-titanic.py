import csv as csv

import numpy as np



csv_file_object = csv.reader(open('../input/train.csv', 'rt')) 

header = next(csv_file_object) 

data=[] 



for row in csv_file_object:

    data.append(row)

data = np.array(data) 



print(type(data[0::,5]))



ages_onboard = data[0::,5].astype(np.float) 
import pandas as pd

import numpy as np



# For .read_csv, always use header=0 when you know row 0 is the header row

df = pd.read_csv('../input/train.csv', header=0)

#df

print(df.head(3))

print(type(df))

print(df.dtypes)

print(df.columns)
print(df.info())
print(df.describe())
print(df['Age'][0:10])
print(type(df['Age']))
print(df['Age'].mean())

print(df['Age'].median())
print(df[ ['Sex', 'Pclass', 'Age'] ])
print(df[df['Age'] > 60])
print(df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']])
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
for i in range(1,4):

    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))
import pylab as P

df['Pclass'].hist()

P.show()

df['Age'].hist()

P.show()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)

P.show()



df['Age'].dropna().hist(bins=16, range=(0,80), alpha = 1)

P.show()
print(df.columns)



df['Gender'] = 4

df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )

df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

df['Emb'] = 5

df['Emb'] = df['Embarked'].dropna().map({'S':0, 'C':1, 'Q':2 }).astype(int)

print(df.head(3))

print(type(df.Emb))
# We know the average [known] age of all passengers is 29.6991176 -- 

# we could fill in the null values with that. But maybe the median would be better? 

# (to reduce the influence of a few rare 70- and 80-year olds?) The Age histogram did seem 

# positively skewed. These are the kind of decisions you make as you create your models in a 

# Kaggle competition.

median_ages = np.zeros((2,3))

print(median_ages)

print(df.columns)
for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

print(median_ages)
df['AgeFill'] = df['Age']

df.head()

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
print(df.columns)



for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]

print(df.head(10))
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)



print(df[ df['Age'].isnull() ])
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df.describe()

print(df.columns)

print(df[df['AgeIsNull'].isnull()])
# Feature Engineering

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

df['Age*Class'].hist()

P.show()

df['FamilySize'].hist()

P.show()

print(df.columns)
df.dtypes
df.dtypes[df.dtypes.map(lambda x: x=='object')]



df.dtypes[df.dtypes.map(lambda x:x=='int')]
print(df.columns)

del df['Sex']

print(df.columns)
df = df.dropna()
train_data = df.values

train_data

print(train_data[0:5,])

print(type(train_data))

print(df.columns)
# Removing all the string objects

del df['PassengerId']

del df['Name']

del df['Ticket']

del df['Cabin']

del df['Embarked']

print(df.columns)
print(df.columns)

print(df.dtypes)

print(df.head(10))

del df['AgeIsNull']

del df['Age']

print(df.dtypes)
print(df.head(10))


# Random forest



# Removing all the string objects

# del df['PassengerId']

# del df['Name']

# del df['Ticket']

# del df['Cabin']

# del df['Embarked']

# del df['Sex']





train_data = df.values

train_data

print(df.columns)

print(df.head(5))

print(df.dtypes)
tst = pd.read_csv('../input/test.csv', header=0)

del tst['PassengerId']

del tst['Cabin']

del tst['Name']



# Gender column

tst['Gender'] = 4

tst['Gender'] = tst['Sex'].map({'female':0, 'male':1}).astype(int)



# Embark

tst['Emb'] = 2

tst['Emb'] = tst['Embarked'].map({'S':0, 'C':1, 'Q':2 }).astype(int)



# AgeFill

age_med = np.zeros((2,3))

tst['AgeFill'] = tst['Age']



for i in range(0,2):

    for j in range(0,3):

        age_med[i,j] = tst[(tst['Gender'] == i) & \

                              (tst['Pclass'] == j+1)]['Age'].dropna().median()

#print(age_med)



for i in range(0,2):

    for j in range(0,3):

        tst.loc[(tst.Gender == i) & \

                (tst.Pclass == j+1) & \

                (tst.Age.isnull()), 'AgeFill'] = age_med[i,j]



# print(tst.head(20))



# Feature engineering

tst['FamilySize'] = tst['SibSp'] + tst['Parch']

tst['Age*Class'] = tst.AgeFill * tst.Pclass



del tst['Sex']

del tst['Age']

del tst['Ticket']

del tst['Embarked']



# Handling null in Fare for tst



fare_med = np.zeros((1,3))



for i in range(0, 1):

    for j in range(0, 3):

        fare_med[i,j] = tst[(tst['Pclass'] == j+1)]['Fare'].dropna().median()

print(fare_med)

print(tst.dtypes)



for i in range(0,1):

    for j in range(0,3):

        tst.loc[(tst['Pclass'] == j+1) & (tst.Fare.isnull()), 'Fare'] = fare_med[i,j]



#print(df.head(5))

#print(tst.head(5))

#tst = tst[['Pclass', 'SibSp', 'Parch', 'Fare','Gender', 'Emb', 'AgeFill', 'FamilySize', 'Age*Class']]
tst.dtypes

print (tst[150:155])
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)



# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])



# Take the same decision trees and run it on the test data

test_data = tst.values

output = forest.predict(test_data)



print(output)





    

    



# print(train_data[0:5,])

# print(type(train_data))

#print(df.columns)