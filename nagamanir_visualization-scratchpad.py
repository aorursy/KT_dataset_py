# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head()
reviews['province'].value_counts().head(10).plot.bar()
# what percent of the total is Californian vintage?

(reviews['province'].value_counts().head(10)/len(reviews)).plot.bar()
#orginal categories

reviews["points"].value_counts().sort_index().plot.bar()
#line chart

reviews["points"].value_counts().sort_index().plot.line()
reviews['points'].value_counts().sort_index().plot.area()
##Histogram for interval variable - price

reviews[reviews['price']<200]['price'].plot.hist()
#skewed data

reviews['price'].plot.hist()
reviews['points'].plot.hist()
reviews['province'].value_counts().head(10).plot.pie()
reviews['province'].value_counts().head(10).plot.pie()



# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
###BiVariate Plotting

reviews[reviews['price'] <100].sample(100).plot.scatter(x='price', y='points')
reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')
##Hex plot

reviews[reviews['price']<100].plot.hexbin(x='price', y='points', gridsize=15)
wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv", index_col=0)

wine_counts.head()
wine_counts.plot.bar(stacked=True)
wine_counts.plot.area()
wine_counts.plot.line()
pd.set_option('max_columns', None)

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

pokemon.head()
pokemon['Type 1'].value_counts().plot.bar()
pokemon['HP'].value_counts().sort_index().plot.line()
pokemon.plot.scatter(x="Attack", y="Defense")
pokemon.plot.hexbin(x="Attack", y="Defense", gridsize=15)
pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]

pokemon_stats_legendary 
pokemon_stats_legendary.plot.bar(stacked=True)
pokemon_stats_by_generation  = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed' ]]

pokemon_stats_by_generation 
pokemon_stats_by_generation.plot.line()