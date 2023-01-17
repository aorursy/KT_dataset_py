import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random

from collections import Counter

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
matches = pd.read_csv('../input/extracted-dataset/matches.csv')

matches.head()
matches.columns
players = pd.read_csv('../input/extracted-dataset/players.csv',encoding='latin1',index_col=0)

players.head()
players.columns
q_matches = pd.read_csv('../input/extracted-dataset/qualifying_matches.csv')

q_matches.head()
q_matches.columns
ranks = pd.read_csv('../input/extracted-dataset/rankings.csv')

ranks.head()
ranks.columns
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

players['country_code'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Countries with most players in WTA")

plt.show()
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

matches['winner_name'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Top 20 players with most wins")

plt.show()
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

matches['loser_name'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Top 20 players with most losses")

plt.show()
players['hand'].value_counts().plot.bar(color={'#8470ff','#3cb371','#ff4500'},figsize=(10,6),title="Handedness of players")

plt.show()
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

matches['winner_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with most wins in WTA")

plt.show()
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

matches['loser_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with most loses in WTA")

plt.show()
top_20 = ranks[ranks['ranking']<=20]

top_20['country_ioc'] = 0

top_20['player_id'] = top_20['player_id'].astype(int)

country = []

for index,row in top_20.iterrows():

    country.append((players.loc[row['player_id']])['country_code'])

    

top_20['country_ioc'] = country
colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

top_20['country_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with highest rankings in WTA")

plt.show()

top_5 = ranks[ranks['ranking']<=5]

top_5['country_ioc'] = 0

top_5['player_id'] = top_5['player_id'].astype(int)

country = []

for index,row in top_5.iterrows():

    country.append((players.loc[row['player_id']])['country_code'])

    

top_5['country_ioc'] = country



colours = list()

for i in range(0,10):

    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))

top_5['country_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 5 countries with highest rankings in WTA")

plt.show()

names = players

names['name'] = players['first_name'] + ' ' + players['last_name']

dom_5 = ranks[ranks['ranking']<=5]

dom_5['name'] = 0

dom_5['player_id'] = dom_5['player_id'].astype(int)

name_list = []

for index,row in dom_5.iterrows():

    name_list.append((names.loc[row['player_id']])['name'])

    

dom_5['name'] = name_list
dom_dict = dict()

for index,row in dom_5.iterrows():

    if row['ranking'] in dom_dict:

        dom_dict[row['ranking']] = row['name'] + '  ' + dom_dict[row['ranking']]

    else:

        dom_dict[row['ranking']] = row['name']

        

for key in dom_dict:

    dom_dict[key] = dom_dict[key].split("  ")

top5dom = dict()

for key in dom_dict:

    d = Counter(dom_dict[key])

    top5dom[key] = d.most_common(3)

    

for key in top5dom:

    print("Top 3 players for WTA Rank {0} :{1}".format(key,top5dom[key]))
climb = players

comp = ranks

climb['rank_diff'] = 0

comp.dropna(subset=['player_id'], inplace = True)

comp['player_id'] = comp['player_id'].astype(int)
rank_diff = list()

for index,row in climb.iterrows():

    temp = comp[comp['player_id'] == index]

    rank_diff.append(temp['ranking'].max()-temp['ranking'].min())
climb['rank_diff'] = rank_diff

climb.sort_values(['rank_diff'],ascending=False,inplace=True)

print("These are the players with biggest climbs in womens tennis history:")

print(climb.head())
mod_age = round(matches['winner_age'])

mod_age = mod_age.dropna()

mod_age = mod_age.astype(int)

mod_age.plot.hist(title="Age at which players are winnig the most")

plt.show()