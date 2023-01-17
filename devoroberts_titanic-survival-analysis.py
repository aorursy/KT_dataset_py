import pandas as pd
from pandas import Series, DataFrame

titanic_df = pd.read_csv('../input/titanic/train.csv')
titanic_df.head()
titanic_df.info()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.countplot('Pclass',data=titanic_df,hue='Sex')
def male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex
titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)
titanic_df[0:10]
sns.countplot('Pclass',data=titanic_df,hue='person')
titanic_df['Age'].hist(bins=70)
mean_age = titanic_df['Age'].mean()
mean_age
titanic_df['person'].value_counts()
fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()
fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()
fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()
titanic_df.head()
deck = titanic_df['Cabin'].dropna()
deck.head()
levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']


cabin_df = cabin_df[cabin_df != 'T']
sns.countplot(x='Cabin',data=cabin_df, palette='summer', order=['A', 'B', 'C', 'D', 'E', 'F'])
titanic_df.head(10)
sns.countplot('Embarked', data=titanic_df, palette='muted', hue='Pclass', order=['C', 'Q', 'S'])
# Who was alone? Who was with family?
titanic_df.head(10)
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone']
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
titanic_df.head()
sns.countplot('Alone', data=titanic_df, palette='Blues')
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})

sns.countplot('Survivor', data=titanic_df, palette='Set1')
sns.factorplot(x='Pclass', y='Survived', data=titanic_df)
sns.factorplot(x='Pclass', y='Survived', data=titanic_df, hue='person')
sns.lmplot('Age', 'Survived', data=titanic_df)
sns.lmplot('Age', 'Survived', data=titanic_df, hue='Pclass', palette='winter')
generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins = generations)
sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins = generations)
sns.lmplot('Age', 'Survived', hue='Embarked', data=titanic_df, palette='winter', x_bins = generations)
sns.lmplot('Age', 'Survived', hue='Alone', data=titanic_df, palette='winter', x_bins = generations)


