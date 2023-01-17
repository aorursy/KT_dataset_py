import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.dtypes
for col in train.columns.tolist():

    print(train[col].value_counts().sort_values(ascending = False).head(10))

    print()
train.isnull().sum()
# Fill age with average.

train.loc[train['Age'].isnull(), 'Age'] = train['Age'][train['Age'].notnull()].mean()



# Fill cabin with null with "missing."

train.loc[train['Cabin'].isnull(), 'Cabin'] = 'missing'
train.loc[train['Cabin'].isnull(), 'Cabin']
train.isnull().sum()
drop_cols = ['PassengerId', 'Name', 'Ticket']

cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

con_cols = ['Age', 'Fare']