# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

csv_file_object = csv.reader(open('../input/train.csv', 'rt'))
header = next(csv_file_object)
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)

# ValueError: could not convert string to float:
#ages_onboard = data[0::,5].astype(np.float)
# pandas

df = pd.read_csv('../input/train.csv', header=0)

df.head(3)
df.dtypes
df.info()
df.describe()
df['Age'][0:10]
# df.Age[0:10] # or
df['Age'].mean()
df[['Sex', 'Pclass', 'Age']]
df[df['Age'] > 60]
df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
for i in range(1, 4):
    print (i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)]))

import pylab as P
df['Age'].hist()
P.show()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
df.head(3)
# better for ML
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df.head(3)
median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

median_ages
df['AgeFill'] = df['Age']

df.head()
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), 'AgeFill'] = median_ages[i,j]

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df.head(3)
df['FamilySize'] = df['SibSp'] + df['Parch']
df.head(3)
df['Age*Class'] = df.AgeFill * df.Pclass
df.head(3)
# df.dtypes[df.dtypes.map(lambda x: x == 'object')]

newDf = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
newDf.head(3)
newDf = newDf.dropna()
newDf.head(3)
train_data = newDf.values
train_data

