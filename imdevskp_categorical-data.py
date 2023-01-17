import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/titanic/train.csv')
df = df[['Sex', 'Embarked']]

df.dropna(axis=0, inplace=True)

df.head()
# it is best for categorical data with ordinal value

# with clear order of hierarchy
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

df['Embarked'] = enc.fit_transform(df['Embarked'])

df.head()
# it will binerize the values into 1 or 0

# yes/no -> 1/0

# true/false -> 1/0



# from sklearn.preprocessing import LabelBinarizer

# lb = LabelBinarizer()

# lb.fit_transform(['yes', 'no', 'no', 'yes'])

  

#     array([[1],

#            [0],

#            [0],

#            [1]])
# Binarize labels in a one-vs-all fashion



# pd.get_dummies(df['Sex'], prefix='gender', drop_first=True)
dummies = pd.get_dummies(df['Sex'])

df = pd.concat([df, dummies], axis=1)

df.head()