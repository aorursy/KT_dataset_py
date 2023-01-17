import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
train = pd.read_csv('../input/train.csv')

train.shape
test = pd.read_csv('../input/test.csv')

test.shape
train.head()
train.info()
train.describe()
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
sns.heatmap(test.isnull(), yticklabels=False, cbar=False)
train['Survived'].value_counts()
sns.countplot(data=train, x='Survived')
pd.crosstab([train['Survived']], [train['Pclass']]).style.background_gradient(cmap='summer_r')
train['Title'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

train['Title'].value_counts()
pd.crosstab([train['Survived']], [train['Title']]).style.background_gradient(cmap='summer_r')
train_age = train[train['Age'].notnull()]
train_age.shape
sns.distplot(train_age[train_age['Survived'] == 1]['Age'], hist=False, rug=True, label='Survived')

sns.distplot(train_age[train_age['Survived'] == 0]['Age'], hist=False, rug=True, label='Not')

plt.xticks([0,5,8,9,10,20,40,60,80])
pd.crosstab([train['Survived']], [train['Sex']]).style.background_gradient(cmap='summer_r')
pd.crosstab([train['Survived']], [train['Embarked']]).style.background_gradient(cmap='summer_r')
sns.factorplot(data=train, x='Pclass', y='Survived', hue='Sex')
pd.crosstab([train['Sex'], train['Survived']], train['Pclass']).style.background_gradient(cmap='summer_r')
sns.factorplot(data=train, x='Embarked', y='Survived', hue='Sex')
sns.factorplot(data=train, x='Embarked', y='Survived', hue='Sex', col='Pclass')
groupby_pes = train.groupby(['Pclass', 'Embarked', 'Sex'])

df_pes = pd.DataFrame(groupby_pes.count()['Survived'])

df_pes.columns = ['Count']

df_pes = df_pes.join(groupby_pes.sum()['Survived'])

df_pes
train['Familysize'] = train[['SibSp', 'Parch']].apply(lambda x: x[0] + x[1] + 1, axis=1)
sns.factorplot(data=train, x='Familysize', y='Survived')
sns.factorplot(data=train, x='Familysize', y='Survived', hue='Sex')
train['Fare'].isnull().sum()
train[train['Pclass'] == 1]['Fare'].plot.kde(color='green')

train[train['Pclass'] == 2]['Fare'].plot.kde(color='yellow')

train[train['Pclass'] == 3]['Fare'].plot.kde(color='red')
train[train['Embarked'] == 'S']['Fare'].plot.kde(color='green')

train[train['Embarked'] == 'C']['Fare'].plot.kde(color='yellow')

train[train['Embarked'] == 'Q']['Fare'].plot.kde(color='red')
train[(train['Pclass'] == 1) & (train['Survived'] == 1)]['Fare'].plot.kde(color='green')

train[(train['Pclass'] == 1) & (train['Survived'] == 0)]['Fare'].plot.kde(color='red')
train[(train['Pclass'] == 2) & (train['Survived'] == 1)]['Fare'].plot.kde(color='green')

train[(train['Pclass'] == 2) & (train['Survived'] == 0)]['Fare'].plot.kde(color='red')
train[(train['Pclass'] == 3) & (train['Survived'] == 1)]['Fare'].plot.kde(color='green')

train[(train['Pclass'] == 3) & (train['Survived'] == 0)]['Fare'].plot.kde(color='red')
train[train['Survived'] == 1]['Fare'].plot.kde(color='green')

train[train['Survived'] == 0]['Fare'].plot.kde(color='red')
sns.boxplot(data=train, x='Pclass', y='Fare', hue='Survived')
fig, ax = plt.subplots(1, 2, figsize=(18,8))

palette={"male": "blue", "female": "pink"}

sns.boxplot(data=train[train['Survived'] == 1], x='Pclass', y='Fare', hue='Sex', ax=ax[0], palette=palette)

ax[0].set_title('Survived')

sns.boxplot(data=train[train['Survived'] == 0], x='Pclass', y='Fare', hue='Sex', ax=ax[1], palette=palette)

ax[1].set_title('Did not Survive')

ax[1].set_ylim([0,600])

ax[0].set_ylim([0,600])
fig, ax = plt.subplots(1, 2, figsize=(18,8))

palette={1: "green", 0: "red"}

sns.boxplot(data=train[train['Sex'] == 'male'], x='Pclass', y='Fare', hue='Survived', ax=ax[0], palette=palette)

ax[0].set_title('Male')

sns.boxplot(data=train[train['Sex'] == 'female'], x='Pclass', y='Fare', hue='Survived', ax=ax[1], palette=palette)

ax[1].set_title('Female')

ax[1].set_ylim([0,600])

ax[0].set_ylim([0,600])
train.info()
pd.DataFrame({

    'Cols': train.columns,

    'Remarks': [

        '-',

        'Target Variable',

        'Looks good',

        'Title might be useful',

        'Looks good',

        'Bucketing might be helpful',

        'Combined as family size',

        'Combined as family size',

        '-',

        'Not significant. Confounding variable Pclass',

        '-',

        'Looks good',

        'May be useful',

        'Looks good'

    ]

})