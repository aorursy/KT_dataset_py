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
df  = pd.read_csv('../input/train.csv', header=0)

df_test = pd.read_csv('../input/test.csv', header=0)
import pylab as P

df['Age'].hist()

P.show()
df['Gender'] = 4



df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

df_test['Gender'] = df_test['Sex'].map( {'female': 0, 'male': 1}).astype(int)

df.head(4)
median_ages = np.zeros((2,3))

for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

median_ages
df['AgeFill']  = df['Age']

df_test['AgeFill']  = df_test['Age']
df[ df['Age'].isnull()][['Gender', 'Pclass', 'Age','AgeFill']].head(10)
for i in range(0,2):

    for j in range(0,3):

        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass ==j+1), \

              'AgeFill'] = median_ages[i,j]

        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass ==j+1), \

              'AgeFill'] = median_ages[i,j]
df[ df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)
df_test[ df_test['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(100)
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)

df[df['Age'].isnull()].head(3)
#family size

df['FamilySize'] = df['SibSp'] + df['Parch']

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass
df['Age*Class'].hist()

P.show()
df.dtypes
df_test.dtypes
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df = df.drop(['Name','Sex','Ticket','Cabin','Embarked'], axis =1)

df_test = df_test.drop(['Name','Sex','Ticket','Cabin','Embarked'], axis =1)
df_test.describe()
df = df.drop(['Age'], axis=1)

df_test = df_test.drop(['Age'], axis=1)
df_test[df_test['Fare'].isnull()] = 35
pd.isnull(df_test).sum() > 0

test_data = df_test.values

train_data = df.values
df.describe()
df_test.describe()
np.any(np.isnan(test_data))
np.any(np.isnan(train_data))
train_data.shape
y = train_data[0::,1]

train_data = np.delete(train_data,1,1)
train_data.shape
test_data.shape
y.shape
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data, y)

output = forest.predict(test_data)
output.shape
ids = df_test['PassengerId'].values

predictions_file = open("myfirstforest.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(ids, output))

predictions_file.close()