import pandas as pd

titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_df.head()
# Exploring the data using pandas methods : 'shape', 'info', 'describe', 'dtype', 'mean()', ...
print(f"DataFrame shape : {titanic_df.shape}\n=================================")
print(f"DataFrame info : {titanic_df.info()}\n=================================")
print(f"DataFrame columns : {titanic_df.columns}\n=================================")
print(f"The type of each column : {titanic_df.dtypes}\n=================================")
print(f"How much missing value in every column : {titanic_df.isna().sum()}\n=================================")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="ticks")
plt.style.use("fivethirtyeight")
# Let's first check gender
# 'catplot()': Figure-level interface for drawing categorical plots onto a FacetGrid.
sns.catplot('Sex', data=titanic_df, kind='count')
# Now let separate the gender by classes passing 'Sex' to the 'hue' parameter
sns.catplot('Pclass', data=titanic_df, hue='Sex', kind='count')
# Create a new column 'Person' in which every person under 16 is child.

titanic_df['Person'] = titanic_df.Sex
titanic_df.loc[titanic_df['Age'] < 16, 'Person'] = 'Child'
# Checking the distribution
print(f"Person categories : {titanic_df.Person.unique()}\n=================================")
print(f"Distribution of person : {titanic_df.Person.value_counts()}\n=================================")
print(f"Mean age : {titanic_df.Age.mean()}\n=================================")
sns.catplot('Pclass', data=titanic_df, hue='Person', kind='count')
# visualizing age distribution
titanic_df.Age.hist(bins=80)
# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(titanic_df, hue="Sex", aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()
# We could have done the same thing for the 'person' column to include children:

fig = sns.FacetGrid(titanic_df, hue="Person",aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()
# Let's do the same for class by changing the hue argument:

fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()
# visualizing the dataset again
titanic_df.head()
# First we'll drop the NaN values and create a new object, deck
deck = titanic_df['Cabin'].dropna()
deck
# let's grab that letter for the deck level with a simple for loop
levels = []
for level in deck:
    levels.append(level[0])

cabin_df = pd.DataFrame(levels)
cabin_df.columns = ['Cabin']
cabin_df.sort_values(by='Cabin', inplace=True)
sns.catplot('Cabin', data=cabin_df, kind='count', palette='winter_d')
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot('Cabin', data=cabin_df, kind='count', palette='summer')
titanic_df.head()
# Now we can make a quick factorplot to check out the results, note the 
# order argument, used to deal with NaN values

sns.catplot('Embarked', data=titanic_df, hue='Pclass', kind='count', order=['C', 'Q', 'S'])
titanic_df.head()
# Let's start by adding a new column to define alone
# We'll add the parent/child column with the sibsp column

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df.Alone
# Look for > 0 or == 0 to set alone status
titanic_df.loc[titanic_df['Alone'] > 0, 'Alone'] = 'with Family'
titanic_df.loc[titanic_df['Alone'] == 0, 'Alone'] = 'Alone'
# Let's check to make sure it worked
titanic_df.head()
# Now let's get a simple visualization!
sns.catplot('Alone', data=titanic_df, kind='count', palette='Blues', 
            order=['Alone', 'with Family'])
# Let's start by creating a new column for legibility purposes through mapping
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No', 1:'Yes'})

# Let's just get a quick overall view of survied vs died. 
sns.catplot('Survivor', data=titanic_df, kind='count')
# Let's use a factor plot again, but now considering class
sns.catplot('Pclass', 'Survived', data=titanic_df, kind='point')
# Let's use a factor plot again, but now considering class and gender
sns.catplot('Pclass', 'Survived', data=titanic_df, hue='Person', kind='point')
# Let's use a linear plot on age versus survival
sns.lmplot('Age', 'Survived', data=titanic_df)
# Let's use a linear plot on age versus survival using hue for class seperation
sns.lmplot('Age', 'Survived',hue='Pclass', data=titanic_df)
# Let's use a linear plot on age versus survival using hue for class seperation
generations = [10, 20, 40, 60, 80]
sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df, palette='winter', x_bins=generations)
sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins=generations)