import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

plt.rcParams['figure.figsize'] = (10,6)

plt.rcParams['font.size'] = 14 

sns.set_style('whitegrid')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/titanic/train.csv')
# examine first few rows 

df.head()
# shape of dataset 

df.shape 
# columns 

df.columns 
# info 

df.info() 
# summary statistics

df.describe() 
# summary statistics: transpose it 

df.describe().T
# numerical counts 

df.isnull().sum()
# visualization of missing values 

sns.heatmap(df.isnull(), cmap='viridis')
# gender: numerical summary 

df['Sex'].value_counts()
#  gender: % 

df['Sex'].value_counts(normalize=True)
# gender: visual representation

sns.countplot(x='Sex', data=df)
# gender: separate the gender by class 

sns.countplot(x='Sex', hue='Pclass', data=df)
# let's separate age by child, male, female 

def male_female_child(passenger): 

    # depending on age and sex 

    age, sex = passenger

    if age < 16: 

        return 'child'

    else: 

        return sex 
# create a new column 

df['Person'] = df[['Age','Sex']].apply(male_female_child,axis=1)
# now take a look 

df.head(10) 
# numerical summary of Person 

df['Person'].value_counts() 
# numerical summary of Person 

df['Person'].value_counts(normalize=True) 
# graphical summary of Person

sns.countplot(x='Person', hue='Pclass', data=df)
# average age 

df['Age'].mean()
# age distribution

df['Age'].hist(bins=50)
# age distribution depending on `Sex` 

fig = sns.FacetGrid(df, hue='Sex', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

fig.set(xlim=(0, df['Age'].max()))

fig.add_legend()
# age distribution depending on 'Person' 

fig = sns.FacetGrid(df, hue='Person', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

fig.set(xlim=(0, df['Age'].max()))

fig.add_legend()
# age distribution depending on `Pclass` 

fig = sns.FacetGrid(df, hue='Pclass', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

fig.set(xlim=(0, df['Age'].max()))

fig.add_legend()
# select cabin column 

df['Cabin'].head() 
# drop missing values 

deck = df['Cabin'].dropna()
# check deck 

deck.head() 
# extract only first letter 

levels = [] 

for level in deck: 

    levels.append(level[0])

    

# create cabin df 

cabin_df = pd.DataFrame(levels)

cabin_df.columns = ['Cabin'] 



# examine first few lines 

cabin_df.head()
# plot cabin

sns.countplot(x='Cabin', data=cabin_df, palette='viridis')
# T doesn't make sense! so let's remove it 

cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot(x='Cabin', data=cabin_df, palette='viridis')
# now take a look at dataset again

df.head() 
# let's analyze this question 

df['Embarked'].value_counts() 
# visualization

sns.countplot(x='Embarked', data=df)
# separate by Sex 

sns.countplot(x='Embarked', data=df, hue='Sex')
# separate by Person

sns.countplot(x='Embarked', data=df, hue='Person')
# separate by Pclass 

sns.countplot(x='Embarked', data=df, hue='Pclass')
# create a df 'Alone'

df['Alone'] = df.Parch + df.SibSp

df['Alone']
# > 0 or == 0 to set alone status 

df['Alone'].loc[df['Alone'] > 0 ] = 'With Family'

df['Alone'].loc[df['Alone'] == 0] = 'Alone'
df.head(10) 
# numerical summary of Alone 

df['Alone'].value_counts() 
# numerical summary of Alone 

df['Alone'].value_counts(normalize=True) 
# graphical representation

sns.countplot(x='Alone', data=df)
# 0 == Survived, 1 == Not Survived 

# numerical summary of Survived 

df['Survived'].value_counts() 
# % of Survived 

df['Survived'].value_counts(normalize=True) 
# visualizations 

sns.countplot(x='Survived', data=df, palette='Set2')
# depending on Sex 

sns.countplot(x='Survived', data=df, hue = 'Sex')
# depending on Person 

sns.countplot(x='Survived', data=df, hue='Person')
# depending on Pclass 

sns.countplot(x='Survived', data=df, hue='Pclass')
# divide `Pclass` depending on `Person`  column

sns.catplot(x='Person', y='Survived', col='Pclass', kind='bar', data=df)
# divide `Sex` depending on `Person`  column

sns.catplot(x='Sex', y='Survived', col='Pclass', kind='bar', data=df)
# Did age matter in general?

sns.lmplot('Age', 'Survived', data=df)
sns.lmplot('Age', 'Survived', data=df, hue='Pclass', palette='winter')
# generations 

generations = [10, 20,40, 60, 60]

sns.lmplot('Age', 'Survived', data=df, hue='Pclass', palette='winter', x_bins=generations)
generations = [10, 20,40, 60, 60]

sns.lmplot('Age', 'Survived', data=df, hue='Sex', palette='winter', x_bins=generations)