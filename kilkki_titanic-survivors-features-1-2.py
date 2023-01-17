import numpy as np

import pandas as pd

from pylab import *



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv', dtype={'Survived': bool})

train.head()
train.groupby('Survived').count()
train.groupby('Sex')['Survived'].mean().plot(kind='bar')

gca().set_ylim((0, 1))

gca().set_ylabel('Sample survival probability')
round_age = (train['Age']/5).round()*5 + 2.5

train.where(train['Sex']=='female').groupby(round_age)['Survived'].mean().plot(style='_', mew=3)

train.where(train['Sex']=='male').groupby(round_age)['Survived'].mean().plot(style='_', mew=3)

gca().set_xlim((0, 85))

legend(['female', 'male'], loc='best')

((train.Age>60) & (train.Sex=='female')).sum()
train.Cabin.unique()
# Extract deck information from cabin column

train['Deck'] = train['Cabin'].apply(lambda s: s[0] if type(s)==str else 'N/A')



# Association of deck and survival

survived_by_deck = train['Survived'].groupby(train['Deck'])



# Standard error of the mean to get a sense of how significant the differences are

standard_error = survived_by_deck.std()/survived_by_deck.count()**.5



# 

survived_by_deck.mean().plot(yerr=standard_error, kind='bar')

gca().set_ylabel('Sample survival probability')
# Names probably signify something, like nationality. 

# Extract title from name.

name_to_title = lambda s: next(ss[:-1] for ss in s.split() if ss.endswith('.'))

train['Title'] = train['Name'].apply(name_to_title)



# Reduce dimensions of title. The removed categories are all rare.

train.loc[train['Title'].isin(['Don', 'Dr', 'Master', 'Rev', 'Jonkheer', 'Sir', 'Countess']), 'Title'] = 'HighStatus'

train.loc[train['Title'].isin(['Capt', 'Col', 'Major']), 'Title'] = 'Officer'

train.loc[train['Title'].isin(['Lady', 'Mlle', 'Ms']), 'Title'] = 'Miss'

train.loc[train['Title'].isin(['Mme']), 'Title'] = 'Mrs'



# Plot association with survival

survived_by_title = train['Survived'].groupby(train['Title'])

means = survived_by_title.mean()

standard_error = survived_by_title.std()/survived_by_title.count()**.5

means.plot(kind='bar', yerr=standard_error)

gca().set_ylabel('Sample survival probability')

# Ticket fare is a heavy-tailed variable. 

# I suspect its logarithm behaves more nicely.

train.loc[train['Fare']==0, 'Fare'] = np.nan

train['LogFare'] = train['Fare'].apply(np.log)



# Association to survival

counts_by_fare = train.groupby(['Survived', train['LogFare'].round()])['LogFare'].count()

hist([

    train['LogFare'].where(train['Survived']).dropna(), 

    train['LogFare'].where(~train['Survived']).dropna(),], 

    stacked=True)

legend(['Survived', 'Didn\'t survive'])

gca().set_xlabel('Log-Fare')

gca().set_ylabel('Count')
# Whether the age is estimated is strongly indicative of not surviving in the sample. 

# Maybe worth including.

# The metadata says it's estimated if its value ends in ".5"

# Include "whether estimated" as a variable

train['AgeIsEstimated'] = (train['Age']==train['Age'].apply(np.floor)+.5).astype(float)



# Association to survival (rather strong!)

groups = train.groupby('AgeIsEstimated')['Survived'].mean()

groups.plot(kind='bar')