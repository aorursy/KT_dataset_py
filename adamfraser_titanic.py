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
df = pd.read_csv('../input/train.csv')

df.head()
type(df)
df.dtypes
df.info()
df.describe()
df['Age'][0:10]
df.Age[0:10]
df['Cabin']
type(df['Age'])
df['Age'].mean()
df['Age'].median()
df[['Sex', 'Pclass', 'Age']]
df[df['Age'] > 60]
df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
for i in range(1,4):

    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))
df['Gender'] = 4

df.head()
df['Gender'] = df['Sex'].map( lambda x: x[0].upper())

df.head()
df['Gender'] = df['Sex'].map( {'female' : 0, 'male' : 1} )

df.head()
median_ages = np.zeros((2,3))

median_ages
for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

        

median_ages
df['AgeFill'] = df['Age']

df.head()
df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
for i in range(0,2):

    for j in range(0,3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \

               'AgeFill'] = median_ages[i,j]

        

df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
df['AgeIsNull'] = df['Age'].isnull().astype(int)

df.head()
df.describe()
df['FamilySize'] = df['SibSp'] + df['Parch']

df.head()
df['Age*Class'] = df['Age'] * df['Pclass']

df.head()
df.dtypes[df.dtypes.map( lambda x: x == 'object')]
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

df.head()
train_data = df.values

train_data
def clean_data(df):

    

    # Convert sex to binary value

    df['Gender'] = df['Sex'].map( {'female' : 0, 'male' : 1} )

    

    # Replace missing values for age with median value for sex and class

    median_ages = np.zeros((2,3))

    for i in range(0,2):

        for j in range(0,3):

            median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()        

    df['AgeFill'] = df['Age']

    for i in range(0,2):

        for j in range(0,3):

            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \

               'AgeFill'] = median_ages[i,j]

            

    # Replace missing values for fare with median value for class

    median_fares = np.zeros((1,3))

    for i in range(0,3):

        median_fares[0,i] = df[df['Pclass'] == j+1]['Fare'].dropna().median()        

    df['FareFill'] = df['Fare']

    for i in range(0,3):

        df.loc[ (df.Fare.isnull()) & (df.Pclass == j+1), 'FareFill'] = median_fares[0,j]

            

    # Create column for whether age value was missing

    df['AgeIsNull'] = df['Age'].isnull().astype(int)

    

    # Create column for family size

    df['FamilySize'] = df['SibSp'] + df['Parch']

    

    # Create column for product of age and class

    df['Age*Class'] = df['AgeFill'] * df['Pclass']

    

    # Drop unused columns

    df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)

    

    # Convert to array

    data = df.values

    

    return data
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



raw_data = pd.read_csv('../input/train.csv')

clean_data = clean_data(raw_data)



X_train, X_test, Y_train, Y_test = train_test_split(clean_data[0::,1::], clean_data[0::,0])



forest = RandomForestClassifier(n_estimators=100)

forest.fit(X_train, Y_train)

score = forest.score(X_test, Y_test)

score