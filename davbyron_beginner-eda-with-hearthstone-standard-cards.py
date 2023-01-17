import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns', None) # show all DF columns
cards = pd.read_csv('../input/hearthstone-standard-cards/hearthstone_standard_cards/hearthstone_standard_cards.csv')

cards.head()
cards.info()
cards.describe()
cards.nunique()
# show correlation betwen columns



fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(data=cards.corr(), ax=ax, cmap='twilight_shifted', annot=True)



# fix x ticks

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

plt.setp(ax.get_ylabel(), rotation=-90)



# set colorbar

cbar = ax.collections[0].colorbar

cbar.set_label('Correlation between columns in card data', rotation=-90, labelpad=20)



plt.show()

plt.close()
# distribution of card classes



class_names = pd.read_csv('../input/hearthstone-standard-cards/hearthstone_standard_cards/metadata/classes.csv')

classes = cards.groupby('classId').count()

# add in multiclass cards that werent counted in classId

for row in cards['multiClassIds']:

    if len(row) > 2:

        row = row.strip('[]')

        row = row.replace(' ', '')

        row = row.split(',')

        classes.loc[int(row[1]), 'id'] += 1

classes = classes['id'].reset_index()

merged = pd.merge(class_names, classes.rename(columns={'id': 'count', 'classId': 'id'}))



palette = ['#034732ff', '#a68653', '#008148ff', '#4e71f2', '#ebc73b', '#c9e5f5', '#5c5558', '#27159e', '#812ab8', '#9e2a2bff', '#c7c7c7']

merged.plot.pie(y='count',\

                figsize=(7.5,7.5),\

                labels=merged['name'],\

                colors=palette,\

                autopct='%0.2f%%',\

                ylabel='',\

                legend=None,\

                title='Distribution of card classes')

plt.show()

plt.close()
# distribution of card rarity



rarity_names = pd.read_csv('../input/hearthstone-standard-cards/hearthstone_standard_cards/metadata/rarities.csv')

rarities = cards.groupby('rarityId').count()

rarities = rarities['id'].reset_index()

merged = pd.merge(rarity_names, rarities.rename(columns={'id': 'count', 'rarityId': 'id'}))



palette = ['#c7c7c7', '#f5f5f5', '#0033ff', '#8d32e3', '#f7a814']

merged.plot.pie(y='count',\

                figsize=(7.5,7.5),\

                labels=merged['name'],\

                colors=palette,\

                autopct='%0.2f%%',\

                ylabel='',\

                legend=None,\

                title='Distribution of card rarities')



plt.show()

plt.close()
# distribution of minon types



miniontype_names = pd.read_csv('../input/hearthstone-standard-cards/hearthstone_standard_cards/metadata/minionTypes.csv')

miniontypes = cards.groupby('minionTypeId').count()

miniontypes = miniontypes['id'].reset_index()

merged = pd.merge(miniontype_names, miniontypes.rename(columns={'id':'count', 'minionTypeId': 'id'}))



palette = ['#21c437', '#68109e', '#9fa7b5', '#6610f2', '#1d6930', '#8c703b', '#c7d7f0', '#d94214']

merged.plot.pie(y='count',\

                figsize=(7.5,7.5),\

                labels=merged['name'],\

                autopct='%0.2f%%',\

                ylabel='',\

                colors=palette,\

                legend=None,\

                title='Distribution of minion types')



plt.show()

plt.close()
# distribution of card health



health = cards.groupby('health').count()

health = health['id'].reset_index()

health = health.rename(columns={'id':'count'})

health = health.astype('int64')



fig, ax = plt.subplots(figsize=(10, 6))



sns.barplot(data=health, x='health', y='count', ax=ax, color='red')

ax.set_title('Distribution of card health')

plt.show()

plt.close()
# distribution of card attacks



attacks = cards.groupby('attack').count()

attacks = attacks['id'].reset_index()

attacks = attacks.rename(columns={'id':'count'})

attacks = attacks.astype('int64')



fig, ax = plt.subplots(figsize=(10, 6))



sns.barplot(data=attacks, x='attack', y='count', ax=ax, color='yellow')

ax.set_title('Distribution of card attacks')

plt.show()

plt.close()
# distribution of mana cost



mana = cards.groupby('manaCost').count()

mana = mana['id'].reset_index()

mana = mana.rename(columns={'id':'count'})



fig, ax = plt.subplots(figsize=(10, 6))



sns.barplot(data=mana, x='manaCost', y='count', ax=ax, color='blue')

ax.set_title('Distribution of card mana cost')

plt.show()

plt.close()
# let's compare health and attack

fig, ax = plt.subplots(figsize=(10, 6))



health_attack = cards.groupby(['health', 'attack'])['id'].count().reset_index()

health_attack = health_attack.astype('int64')



sns.set_style('darkgrid')

sns.scatterplot(data=health_attack, x='attack', y='health', size='id', sizes=(20, 4000), ax=ax, alpha=0.4, legend=False)

ax.set_title('Frequency distribution of cards for each attack/health combination')



plt.show()

plt.close()
# mana cost distribution by class



fig, ax = plt.subplots(figsize=(10, 6))



class_mana_count = cards.groupby(['classId', 'manaCost'])['id'].count().reset_index()

merged = pd.merge(class_names, class_mana_count.rename(columns={'id':'count', 'classId':'id'}))



sns.set_style('white')

palette = ['#034732ff', '#a68653', '#008148ff', '#4e71f2', '#ebc73b', '#c9e5f5', '#5c5558', '#27159e', '#812ab8', '#9e2a2bff', '#c7c7c7']

sns.scatterplot(data=merged, x='name', y='manaCost', size='count', sizes=(20, 4000), alpha=0.6, legend=False, ax=ax, hue='name', palette=palette)

ax.set_yticks(range(11))

plt.xticks(rotation=45)

ax.set_xlabel('class')

ax.set_ylabel('mana cost')

ax.set_title('Frequency distribution of card mana cost per class')



plt.show()

plt.close()