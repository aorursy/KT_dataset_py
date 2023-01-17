import numpy as np

import pandas as pd
train_data = pd.read_csv('../input/train.csv', header=0)

test_data = pd.read_csv('../input/test.csv', header=0)

test_data.loc[:, 'Survived'] = None



frames = [train_data, test_data]

df = pd.concat(frames, axis=0, join='outer', ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False, copy=True) # ignore_index=True
df.tail(5)
df.shape
# df.info() can id missing values

df.info()
df.describe()
# check null values by column

df.isnull().sum(0)
df.groupby(['Pclass']).size()
df[df['Survived'] == 1].groupby(['Pclass']).size()
df[df['Sex'] == 'female'].groupby(['Pclass']).size()
df[(df['Survived'] == 1) & (df['Sex'] == 'female')].groupby(['Pclass']).size()
# histogram in python with pylab

import pylab as pl

df['Age'].hist() # only generates the object

pl.show()
df['Age'].dropna().hist(bins=20, range=(0, 80), alpha=.5)

pl.show()
# data manipulation

df['gender'] = df['Sex'].map( lambda x: x[0].lower() ) 



df.head(5)
df['gender_num'] = df['Sex'].map( {'female':0, 'male':1} ).astype(int)

df.dtypes
df['Age'].dropna().map( lambda x: round(x)).astype(int).min()

df['Age'].dropna().map( lambda x: round(x)).astype(int).max()

df['Age'].dropna().map( lambda x: round(x)).astype(int).values

age_range = df['Age'].max() - df['Age'].min()
df['age_fill'] = df['Age']
# generate a median age table to use for missing data imputation



median_ages = np.zeros((2, 3)).astype(float)

median_ages
df['age_fill'].isnull().sum(0) # 177 rows have NaN
class_range = df['Pclass'].max() - df['Pclass'].min() 

gender_range = df['gender_num'].max() - df['gender_num'].min()



for i in range(0, gender_range + 1):

        for j in range(0, class_range + 1):

            median_ages[i, j] = df[(df['gender_num'] == i) & (df['Pclass'] == (j + 1))]['Age'].median()

            df.loc[ (df['Age'].isnull() & (df['Pclass'] == (j + 1)) & (df['gender_num'] == i)), 'age_fill' ] = median_ages[i, j]

            

median_ages

df['age_fill'].isnull().sum(0) # all filled
import pylab as pl

df['age_fill'].hist()

#df['Age'].hist()

pl.show()
df[ df['Age'].isnull() ][ ['Age', 'Pclass', 'age_fill', 'Name', 'Sex', 'gender_num'] ].head(10)
df[df.Embarked.isnull()]
df.groupby('Embarked').size()
df['embarked_fill'] = df['Embarked']

df.loc[df.embarked_fill.isnull(), 'embarked_fill'] = 'S'

df['ageIsNull'] = df.Age.isnull().astype(int)
# combo variables

df['age*class'] = df['ageIsNull'] * df['Pclass']



df['group_size'] = df['SibSp'] + df['Parch'] + 1
df.dtypes[df.dtypes.map(lambda x: x == 'object')]
df['Fare'].median()

df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].median()
df_1 = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'gender', 'embarked_fill', 'Age', 'age*class'], axis=1, inplace=False)

df_1.head()
print(df_1.shape)

print(train_data.shape)

print(test_data.shape)
df_1.isnull().sum(0)
print(df_1.isnull().sum(0))

print(df_1.dtypes)
df_1['survival'] = df_1['Survived']

df_1['survival'] = df_1['survival'].apply(pd.to_numeric, errors='coerce')
df_1.drop(['Survived'], axis=1, inplace=True)

df_1.dtypes
all = df_1.values
train = all[:train_data.shape[0], :]

test = all[train_data.shape[0]:df.shape[0], :]

print(train.shape, test.shape)

all[890, :]
from sklearn.ensemble import RandomForestClassifier 



model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

predictions = model.fit(np.delete(train, 8, axis=1), list(train[:, 8]))
# Take the same decision trees and run it on the test data

output = predictions.predict(np.delete(test, 8, axis=1))
final = pd.DataFrame({'PassengerId': test_data.loc[:, 'PassengerId'], 'Survived': output})
final.tail()
# DataFrame.to_csv()

# final.to_csv("./data/kaggleTitanic21py.csv", index=False)
import time

print(time.ctime())