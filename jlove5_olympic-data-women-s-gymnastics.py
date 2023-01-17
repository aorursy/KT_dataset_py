# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read the data into a dataframe

raw_data = pd.read_csv('../input/athlete_events.csv')

noc = pd.read_csv('../input/noc_regions.csv')
raw_data.head()
raw_data['Sport'].unique()
raw_data['Games'].unique()
#Filter out the gymnastics events. This dataset has both men's and women's gymnastics.

df = raw_data.loc[raw_data['Sport']=='Gymnastics']
df.head()
df.tail()
df['Event'].unique()
#Filter out the women's events: I'm just interested in Women's Artistic Gymnastics

women_events = ['Gymnastics Women\'s Individual All-Around', 'Gymnastics Women\'s Team All-Around', 'Gymnastics Women\'s Horse Vault', 'Gymnastics Women\'s Floor Exercise',

               'Gymnastics Women\'s Uneven Bars', 'Gymnastics Women\'s Balance Beam']

womens = df.loc[df['Event'].isin(women_events)]
womens['Year'].unique()
#Replace Event names to something less wordy

womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Individual All-Around', 'All-Around')
womens.head()
#Write a loop or function for this 

womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Team All-Around', 'Team')

womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Horse Vault', 'Vault')

womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Balance Beam', 'Balance Beam')

womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Uneven Bars', 'Uneven Bars')
womens['Event'] = womens['Event'].str.replace('Gymnastics Women\'s Floor Exercise', 'Floor')
womens.head()
womens.tail()
womens['Event'].unique()
#Keep only medal winners

womens = womens.dropna(subset=['Medal'])
womens.head()
womens['Event'].unique()
womens['Year'].unique()
g = sns.catplot(data=womens, kind='count', x='Team', order = womens['Team'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

g.fig.suptitle("Total number of Olympic medals in Women's Artistic Gymnastics: 1928-2016")

a = womens['Team'].value_counts().to_frame()

a.iloc[0:10]
womens.head()
sns.set_palette('Blues')

beam = womens.loc[womens['Event']=='Balance Beam',]

beam.reset_index(drop=True, inplace=True)
beam.head()
g = sns.catplot(data=beam, kind='count', x='Team', order = beam['Team'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

g.fig.suptitle("Olympic Balance Beam Medals")

beam['Team'].value_counts().to_frame()
floor = womens.loc[womens['Event']=='Floor',]

floor.reset_index(drop=True, inplace=True)
f = sns.catplot(data=floor, kind='count', x='Team', order = floor['Team'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

f.fig.suptitle("Olympic Floor Exercise Medals")

floor['Team'].value_counts().to_frame()
bars = womens.loc[womens['Event']=='Uneven Bars',]

bars.reset_index(drop=True, inplace=True)
ub = sns.catplot(data=bars, kind='count', x='Team', order = bars['Team'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

ub.fig.suptitle("Olympic Uneven Bars Medals")

bars['Team'].value_counts().to_frame()
vault = womens.loc[womens['Event']=='Vault',]

vault.reset_index(drop=True, inplace=True)



v = sns.catplot(data=vault, kind='count', x='Team', order = vault['Team'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

v.fig.suptitle("Olympic Vault Medals")

vault['Team'].value_counts().to_frame()
sns.set_palette('Reds')

ussr = womens.loc[womens['Team']=='Soviet Union',]

ussr.reset_index(drop=True, inplace=True)
a = sns.catplot(data=ussr, kind='count', x='Medal', order = womens['Medal'].value_counts().index, height=4, aspect=3).set_xticklabels(rotation=90)

a.fig.suptitle("Total number of Olympic medals, USSR, Women's Artistic Gymnastics: 1928-1988")
b = sns.catplot(data=ussr, kind='count', x='Year').set_xticklabels(rotation=90)

b.fig.suptitle("Olympic Medals, USSR")
# Create column: Medals earned percent of total, 18 medals total can be earned

grouped = ussr.groupby('Year')

grouped['Medal'].value_counts().to_frame()

#ussr['Medal'].value_counts()

b = sns.catplot(data=ussr, kind='count', hue='Medal', x='Year').set_xticklabels(rotation=90)

b.fig.suptitle("Olympic Medals, USSR")
ussr.head()
sns.set_palette('Reds')

usa = womens.loc[womens['Team']=='United States',]

usa.reset_index(drop=True, inplace=True)
usa.head()
u = sns.catplot(data=usa, kind='count', x='Year', height=4, aspect=3).set_xticklabels(rotation=90)

u.fig.suptitle('Distribution of Medals')
sns.set_palette('Blues')

teams = womens.loc[womens['Event']=='Team',]

teams.reset_index(drop=True, inplace=True)
t = sns.catplot(data=teams, kind='count', x='Team', height=4, aspect=3, order=teams['Team'].value_counts().index).set_xticklabels(rotation=90)

t.fig.suptitle('Distribution of Team Medals')

teams['Team'].value_counts().to_frame()
ROM = teams.loc[teams['Team']=='Romania',]

ROM.reset_index(drop=True, inplace=True)

ROM.head()
ROM['Year'].unique()
sns.set_palette('dark')

r = sns.catplot(data=ROM, kind='count', x = 'Medal', order=['Gold', 'Silver', 'Bronze'])

r.fig.suptitle('Romania: Team Medals 1956-2012')
g = sns.catplot(data=ROM, kind='count', x='Age', height=4, aspect=3).set_xticklabels(rotation=90)

g.fig.suptitle('Romanian Teams: Age of Gymnasts')
g = sns.catplot(data=womens, kind='count', x='Age', height=4, aspect=3).set_xticklabels(rotation=90)

g.fig.suptitle('WAG Olympic Gymnasts: Age of Gymnast')
womens['Age'].value_counts().to_frame()
c = sns.catplot(data=ussr, kind='count', x='Age', height=4, aspect=3).set_xticklabels(rotation=90)

ussr['Age'].value_counts().to_frame()
modern = womens.loc[womens['Year'].isin(['2008', '2012', '2016'])]
modern.head()
modern['Year'].unique()
g = sns.catplot(data=modern, kind='count', x='NOC', order = modern['NOC'].value_counts().index).set_xticklabels(rotation=90)

g.fig.suptitle("Medal count by Country: Open-ended scoring")

modern['NOC'].value_counts().to_frame()
g = sns.catplot(data=modern, kind='count', x='Age', height=4, aspect=3).set_xticklabels(rotation=90)

modern['Age'].value_counts().to_frame()
grouped_modern = modern.groupby('NOC')

grouped_modern['Event'].value_counts().to_frame()