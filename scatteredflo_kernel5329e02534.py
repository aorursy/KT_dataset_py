# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Series

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



plt.style.use('seaborn')

sns.set(font_scale=2.5) 

# 이 두줄은 본 필자가 항상 쓰는 방법입니다. 

# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno





%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
train.describe()
print(train.shape, test.shape)
test.head()
train
train.isnull().sum()
train.shape
len(train)
train.isnull().sum() / len(train) *100
round(test.isnull().sum() / len(test) * 100)
msno.matrix(df=train, figsize=(8,8), color = (0.8, 0.5, 0.2))
msno.matrix(df=test, figsize=(8,8), color = (0.8, 0.5, 0.2))
train['Embarked'].value_counts()
train['Survived'].mean()
train.head()
train.head()
train.groupby(['Pclass'])['Survived'].count()
train.groupby(['Pclass'])['Survived'].mean().plot.bar()
sns.countplot('Pclass', hue='Survived', data=train)

plt.show()
sns.countplot('Sex', hue='Survived', data=train)

plt.show()
sns.countplot('Embarked', hue='Survived', data=train)

plt.show()
sns.factorplot('Embarked', 'Survived', hue='Sex', data=train, aspect=1.5)
train.groupby(['Sex','Embarked'])['Survived'].mean()
# Age

train['Age'].describe()
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(train['Age'], ax=ax)

plt.show()
train['Survived'] == 1
train[train['Survived'] == 1]['Age']

train[train['Survived'] == 1]['Age']
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(train[train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(train[train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['1', '0'])

plt.show()
train['Pclass'].unique()
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(train[train['Pclass'] == 1]['Age'], ax=ax)

sns.kdeplot(train[train['Pclass'] == 2]['Age'], ax=ax)

sns.kdeplot(train[train['Pclass'] == 3]['Age'], ax=ax)

plt.legend(['1', '2','3'])

plt.show()
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(train[train['Age'] < i]['Survived'].sum() / len(train[train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train.head()
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train.groupby(['Survived'])['FamilySize'].mean()
train_new = pd.DataFrame(round(train.groupby(['FamilySize'])['Survived'].mean(),2))

train_new = train_new.reset_index()

train_new
sns.factorplot('FamilySize', 'Survived', data=train_new, aspect=3)