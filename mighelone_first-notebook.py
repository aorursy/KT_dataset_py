# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# map function
# helpful for mapping categorical variables into int
mapping = lambda x: {label:idx for idx,label in enumerate(np.unique(x))}
mapped_features = {}
# Read train data
train = pd.read_csv('../input/train.csv', index_col='PassengerId')
train.head()
train.describe()
train.info()

train.columns[train.count() < len(train)]
train.dtypes
ax = train['Survived'].value_counts().plot.bar(color='red', alpha=0.7)
ax.set_title('Passenger survived')
ax.set_ylabel('Frequency')
ax = train.Sex.value_counts().plot.bar(color='red', alpha=0.7)
ax.set_title('Passenger sex')
train.Sex.unique()
pd.crosstab(train['Sex'], train['Survived']).plot(kind='bar', stacked=True)
sex_mapping = mapping(train.Sex)
mapped_features['Sex'] = sex_mapping
print(sex_mapping)
train['Sex'] = train['Sex'].map(sex_mapping)
print(train['Sex'].unique())

print('Number of Age samples: {}'.format(train['Age'].count()))
print('Number of sample: {}'.format(len(train['Age'])))
print('Number of null ages: {}'.format(train.Age.isnull().sum()))
train['Age'].count() + train.Age.isnull().sum() == len(train['Age'])
# Age per Sex
print(train.loc[train.Sex == 0, 'Age'].describe())
print(train.loc[train.Sex == 1, 'Age'].describe())
