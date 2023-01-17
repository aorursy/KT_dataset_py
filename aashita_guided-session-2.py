import pandas as pd

import re # For regular expressions
path = '../input/'

df = pd.read_csv(path + 'train.csv') 

df = df.set_index('PassengerId')

df.head()
df['Title'] = df['Name'].map(lambda name: re.findall("\w+[.]", name)[0])



title_dictionary = {'Ms.': 'Miss.', 'Mlle.': 'Miss.', 

              'Dr.': 'Rare', 'Mme.': 'Mr.', 

              'Major.': 'Rare', 'Lady.': 'Rare', 

              'Sir.': 'Rare', 'Col.': 'Rare', 

              'Capt.': 'Rare', 'Countess.': 'Rare', 

              'Jonkheer.': 'Rare', 'Dona.': 'Rare', 

              'Don.': 'Rare', 'Rev.': 'Rare'}



df['Title'] = df['Title'].replace(title_dictionary)



df.head()
df.groupby('Title')
df.groupby('Title').median()
df.groupby('Title').mean()
df['MedianAge'] = df.groupby('Title')['Age'].transform("median")

df.head(15)
df['Age'] = df['Age'].fillna(df['MedianAge'])

df.head()
df = df.drop('MedianAge', axis=1)

df.head()
df.isnull().sum()
df.dtypes
df['Sex'].value_counts()
df = df.replace({'male': 0, 'female': 1})
df.dtypes
df['Embarked'].value_counts()
pd.get_dummies(df['Embarked']).head()
port_df = pd.get_dummies(df['Embarked'], prefix='Port')
port_df.head()
df = pd.concat([df, port_df], axis=1)
df.head()
df = df.drop('Embarked', axis=1)
df.columns