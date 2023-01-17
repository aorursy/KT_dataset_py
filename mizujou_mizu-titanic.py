# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
df_train.info()
df_train.shape, df_test.shape
df_train.columns, df_test.columns
df_train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
df_train.head()
df_train['Sex'].value_counts()
df_train['Sex'] = df_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
df_test['Sex'] = df_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns
import matplotlib.pyplot as plt

age_g = sns.FacetGrid(df_train, col='Survived')
age_g.map(plt.hist, 'Age', bins=10)

plt.show()
df_train['Parch'].unique()
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False)
df_train[["Parch", "Pclass"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False)
age_g = sns.FacetGrid(df_train, col='Survived')
age_g.map(plt.hist, 'Pclass', bins=3)

plt.show()
df_train.info()
df_train.head()
df_train.drop(['Embarked'], axis=1, inplace=True)
df_test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'Embarked'], axis=1, inplace=True)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
df_train_imp = pd.DataFrame(my_imputer.fit_transform(df_train))

# Imputation removed column names; put them back
df_train_imp.columns = df_train.columns
df_train_imp.info()
