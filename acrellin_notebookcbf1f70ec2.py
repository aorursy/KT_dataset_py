# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.info()

print('-' * 30)

test.info()
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)

test = test.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)
train.info()

print()

test.info()
# Fill missing fare value in test df

test['Fare'].fillna(test['Fare'].median(), inplace=True)

test.info()
# Fill missing age values - draw from existing distribution

train_age_mean = train['Age'].mean()

train_age_std = train['Age'].std()



test_age_mean = test['Age'].mean()

test_age_std = test['Age'].std()





# Plot original age dist

sns.distplot(train['Age'][train['Age'].notnull()])

train['Age'][train['Age'].isnull()] = np.random.normal(train_age_mean, train_age_std, train['Age'].isnull().sum())

test['Age'][test['Age'].isnull()] = np.random.normal(test_age_mean, test_age_std, test['Age'].isnull().sum())
sns.distplot(train['Age'][train['Age'].notnull()])
train.info()

print()

test.info()