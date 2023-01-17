from IPython.display import display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/train.csv')
display(df.head())
display(df.describe())
df['Embarked'].value_counts()
df['SibSp'].value_counts()
df['Ticket'].value_counts()[:10]
_ = sns.barplot(data=df, x='Sex', y='Survived')
_ = sns.factorplot(data=df, x='Sex', kind='count', col='Survived')

_ = sns.barplot(data=df, x='Sex', y='Survived', hue='Pclass')
df.loc[: ,'Age'].hist(bins=40)
plt.title('Age Distribution of Passengers')
_ = plt.show()
df.loc[df['Survived'] == 1, 'Age'].hist(alpha=0.5, bins=40, label='Survived')
df.loc[df['Survived'] == 0, 'Age'].hist(alpha=0.5, bins=40, label='Perished')
plt.title('Age Distributions')
plt.legend()
_ = plt.show()
males = df[df['Sex'] == 'male']
males.loc[males['Survived'] == 1, 'Age'].hist(alpha=0.5, bins=40, label='Survived')
males.loc[males['Survived'] == 0, 'Age'].hist(alpha=0.5, bins=40, label='Perished')
plt.title('Age Distributions of Male Passengers')
plt.legend()
_ = plt.show()
df['age_bin'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 25, 45, 80])
females = df[df['Sex'] == 'female']
males= df[df['Sex'] == 'male']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches((20, 4))

sns.barplot(data=df, x='age_bin', y='Survived', ax=ax1)
ax1.set_title('Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax1.get_xticklabels()]

sns.barplot(data=males, x='age_bin', y='Survived', ax=ax2)
ax2.set_title('Male Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax2.get_xticklabels()]

sns.barplot(data=females, x='age_bin', y='Survived', ax=ax3)
ax3.set_title('Female Survival Rates by Age Group')
[tick.set_rotation('vertical') for tick in ax3.get_xticklabels()]

_ = plt.show()
import random
random.sample(list(df['Name'].values), 5)
df['title'] = titles = df['Name'].apply(
    lambda x: x.split(',')[1].split('.')[0] if ',' in x else x)
df['title'].value_counts()
test_df = pd.read_csv('../input/test.csv')
test_df['title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
test_df['title'].value_counts()
sns.barplot(data=df, x='title', y='Survived')
_ = plt.xticks(rotation='vertical')
def map_title(title):
    title = title.strip()
    if title in ['Mr','Mrs', 'Mme', 'Miss', 'Ms', 'Mlle']:
        return 'untitled'
    if title in ['Lady', 'Sir', 'Jonkheer', 'Master', 'the Countess', 'Don', 'Dona']:
        return 'titled'
    if title in ['Dr', 'Major', 'Col', 'Capt']:
        return 'service'
    if title in ['Rev']:
        return 'cloth'
    return title
titles.apply(map_title).value_counts()
df['title_'] = df['title'].apply(map_title)
_ = sns.barplot(data=df, x='title_', y='Survived', hue='Sex')
def map_marriage(title):
    title = title.strip()
    if title in ['Mrs', 'Mme', 'the Countess']:
        return 'married'
    if title in ['Ms', 'Miss', 'Mlle', 'Lady', 'Dona', 'Rev']:
        return 'unmarried'
    return 'unknown'
titles.apply(map_marriage).value_counts()
df['married'] = df['title'].map(map_marriage)
married = ((df['Age'] >= 25) & (df['SibSp'] > 0)).apply(
    lambda x: 'married' if x else 'unmarried')
mask = df['married'] == 'unknown'
df.loc[mask, 'married'] = married[mask]

df['married'].value_counts()
df = df.sort_values(by='married')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches((15, 4))

titles = ['Marriage Status and Survival Rate',
          'Marriage Status and Survival Rate by Sex']
sns.barplot(data=df, x='married', y='Survived', ax=ax1)
ax1.set_title(titles[0])
sns.barplot(data=df, x='Sex', y='Survived', hue='married')
ax2.set_title(titles[1])
_ = plt.show()
_ = sns.factorplot(data=df, x='married', hue='Sex', kind='count')
df['deck'] = df['Cabin'].apply(lambda x: x if pd.isna(x) else x[0])
print("value counts:")
print(df['deck'].value_counts())
print("finite count:")
print(df['deck'].value_counts().sum())
print("nan count:")
print(df['deck'].isna().sum())
_ = sns.barplot(data=df, x='deck', y='Survived', hue='Sex')
_ = sns.barplot(data=df, x='deck', y='Fare')
df['fare_bin'] = pd.cut(df['Fare'], bins=list(range(0, 150, 10)))
_ = df['fare_bin'].value_counts().sort_index().plot(kind='bar')
sns.barplot(data=df, x='fare_bin', y='Survived')
_ = plt.xticks(rotation='vertical')
sns.barplot(data=df, x='fare_bin', y='Survived', hue='Sex')
plt.gcf().set_size_inches((10, 4))
_ = plt.xticks(rotation='vertical')