# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plt.rcParams['figure.figsize'] = [6,3]

plt.rcParams['figure.dpi'] = 80
data = sns.load_dataset('titanic')

print(data.head())

print(data.columns)
from pandas_profiling import ProfileReport

train_profile = ProfileReport(data, title='Pandas Profiling Report', html={'style':{'full_width':True}})

train_profile
data.describe()
data.info()
plt.style.use('ggplot')
# Check null values

data.isnull().sum()
# Plot null values using heatmap

sns.heatmap(data.isnull(), cmap='viridis', cbar=True)
# CHeck correlation of the dataset

corrmat = data.corr()

corrmat
# Plot Heatmap

sns.heatmap(corrmat)
print(list(data))
# Plot the countplot and DIstplot

fig, ax = plt.subplots(3,3, figsize=(16,16))



sns.countplot('survived', data=data, ax=ax[0][0])

sns.countplot('pclass', data=data, ax=ax[0][1])

sns.countplot('sex', data=data, ax=ax[0][2])

sns.countplot('sibsp', data=data, ax=ax[1][0])

sns.countplot('parch', data=data, ax=ax[1][1])

sns.countplot('embarked', data=data, ax=ax[1][2])

sns.countplot('alone', data=data, ax=ax[2][0])



sns.distplot(data['fare'], kde=True, ax=ax[2][1])

sns.distplot(data['age'], kde=True, ax=ax[2][2])

plt.tight_layout()
# Check the number of survived or not value

data['survived'].value_counts()
# Plot the graph number of survived or not using countplot

sns.countplot('survived', data=data)

plt.title('Titanic Survived Plot')

plt.show()
# plot the hist plot

data['survived'].plot.hist()
# Check number of survived or not using pie chart

data['survived'].value_counts().plot.pie()
# Check number of survived or not using pie chart

data['survived'].value_counts().plot.pie(autopct='%1.2f%%')
# Check number of survived or not using pie chart

explode = [0,0.1]

data['survived'].value_counts().plot.pie(explode=explode, autopct='%1.2f%%')
# Check the number of Pclass

data['pclass'].value_counts()
# Groupby

data.groupby(['pclass', 'survived'])['survived'].count()
# check how many passenger travel in 1st claess, 2nd class and 3rd class using countplot 

sns.countplot('pclass', data=data)
# check how many male and female survived in 1st, 2nd and 3rd class using countplot

sns.countplot('pclass', data=data, hue='survived')
# check how many passenger travel in 1st claess, 2nd class and 3rd class using pie chart 

data['pclass'].value_counts().plot.pie(autopct = "%1.1f%%")
sns.catplot(x = 'pclass', y='survived', kind='bar', data=data)
sns.catplot(x='pclass', y='survived', kind='point', data=data)
sns.catplot(x='pclass', y='survived', kind='violin', data=data)
data['sex'].value_counts()
data.groupby(['sex', 'survived'])['survived'].count()
sns.countplot('sex', data=data)
data['sex'].value_counts().plot.pie(autopct = '%1.1f%%')
sns.catplot(x='sex', y='survived',kind='bar',data=data)
sns.catplot(x='sex', y='survived', kind='bar', data=data, hue='pclass')
sns.catplot(x='sex', y='survived', kind='bar', data=data, col='pclass')
sns.catplot(x='sex', y='survived', kind='bar', data=data, row='pclass')
sns.catplot(x='pclass', y='survived', kind='bar', data=data, col='sex')
sns.catplot(x='sex', y='survived', kind='point', data=data)
sns.catplot(x='sex', y='survived', kind='point', data=data,hue='pclass')
sns.catplot(x='pclass', y='survived', kind='point', data=data, hue='sex')
sns.catplot(x='sex', y='survived', kind='violin', data=data)
sns.catplot(x='sex', y='survived', kind='violin', data=data, hue='pclass')
sns.catplot(x='sex', y='survived', kind='violin', data=data, col='pclass')
data['age'].hist(bins=30, density=True, color = 'orange', figsize = (10,8))

plt.xlabel('Age')

plt.show()
sns.distplot(data['age'])
sns.distplot(data['age'], hist=False)
sns.distplot(data['age'], hist=True)
sns.kdeplot(data['age'], shade=True)
sns.catplot(x='sex', y='age', data=data, kind='box')
sns.catplot(x='sex', y='age', data=data, kind='box', hue='pclass')
sns.catplot(x='sex', y='age', data=data, kind='box', col='pclass')
sns.catplot(x='pclass', y='age', data=data, kind='violin')
sns.catplot(x='pclass', y='age', data=data, kind='violin', hue='sex')
sns.catplot(x='pclass', y='age', data=data, kind='violin', col='sex')
sns.catplot(x='pclass', y='age', kind='swarm', data=data)
sns.catplot(x='pclass', y='age', kind='swarm', data=data, col='sex')
sns.catplot(x='survived', y='age', kind='swarm', data=data, col='sex')
sns.catplot(x='survived', y='age', kind='swarm', data=data, row='sex', col='pclass')
data['fare'].hist(bins=40, color='orange')
sns.distplot(data['fare'])

plt.xlabel('fare')

plt.show()
sns.catplot(x='sex', y='fare', data=data, kind='box')
sns.catplot(x='sex', y='fare', data=data, kind='box', hue='pclass')
sns.catplot(x='sex', y='fare', data=data, kind='box', col='pclass')
sns.catplot(x='sex', y='fare', data=data, kind='boxen', col='pclass')
sns.catplot(x='sex', y='fare', data=data, kind='swarm', col='sex')
sns.catplot(x='survived', y='fare', data=data, kind='swarm', col='sex')
sns.catplot(x='survived', y='fare', data=data, kind='swarm', col='pclass')
sns.jointplot(x='age', y='fare', data=data)
sns.jointplot(x='age', y='fare', data=data, kind='kde')
sns.relplot(x='age', y='fare', data=data, row='sex', col='pclass')
data['sibsp'].value_counts()
sns.countplot('sibsp', data=data)
sns.countplot('sibsp', data=data, hue='survived')
sns.catplot(x = 'sibsp', y = 'survived', kind='bar',data=data)
sns.catplot(x = 'sibsp', y = 'survived', kind='bar',data=data, hue='sex')
sns.catplot(x = 'sibsp', y = 'survived', kind='bar',data=data, col='sex')
sns.catplot(x = 'sibsp', y = 'survived', kind='bar',data=data, col='pclass')
sns.catplot(x = 'sibsp', y = 'survived', kind='point',data=data)
sns.catplot(x = 'sibsp', y = 'survived', kind='point',data=data, hue='sex')
sns.catplot(x = 'sibsp', y = 'survived', kind='point',data=data, col='pclass')
sns.catplot(x = 'sibsp', y = 'fare', kind='swarm',data=data, col='sex')
sns.catplot(x = 'sibsp', y = 'fare', kind='swarm',data=data, col='pclass')
data['parch'].value_counts()
sns.countplot('parch', data=data)
sns.countplot('parch', data=data, hue='sex')
sns.catplot(x='parch', y='survived', data=data, kind='bar')
sns.catplot(x='parch', y='survived', data=data, kind='bar', hue='sex')
sns.catplot(x='parch', y='fare', data=data, kind='swarm')
sns.catplot(x='parch', y='fare', data=data, kind='swarm', col='pclass',row='sex')
data['embarked'].value_counts()
sns.countplot('embarked', data=data)
sns.catplot(x='embarked', y='survived', kind='bar', data=data)
sns.catplot(x='embarked', y='survived', kind='bar', data=data, hue='sex')
data.columns
data['who'].value_counts()
sns.countplot('who', data=data)
sns.countplot('who', data=data, hue='survived')
sns.catplot(x='who', y='survived', kind='bar', data=data)
sns.catplot(x='who', y='survived', kind='bar', data=data, hue='pclass')
sns.catplot(x='who', y='survived', kind='bar', data=data, col='pclass')