import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head()
reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')
import matplotlib.pyplot as plt

import numpy as np

#desha = np.array(reviews['country'])

#print(type(desha), desha.shape)

#desha = desha.reshape(1, 150930)

#print(type(desha), desha.shape)

#c = pd.Series(data=desha)

plt.scatter(x=reviews['price'], y=reviews['points'])

plt.title("Scatter plot of price vs points")

plt.xlabel("price")

plt.ylabel("points")
reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')
reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",

                          index_col=0)
wine_counts.head()
wine_counts.plot.bar(stacked=True)
wine_counts.plot.area()
wine_counts.plot.line()
from IPython.display import HTML

HTML("""

<ol>

<li>Scatter plots and hex plots work best with a mixture of ordinal categorical and interval data.</li>

<br/>

<li>Nominal categorical data makes sense in a (stacked) bar chart, but not in a (bivariate) line chart.</li>

<br/>

<li>Interval data makes sense in a bivariate line chart, but not in a stacked bar chart.</li>

<br/>

<li>One way to fix this issue would be to sample the points. Another way to fix it would be to use a hex plot.</li>

</ol>

""")
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

pokemon.head()
pokemon.describe()
pokemon.plot.scatter(x='Attack', y='Defense')
#My code

pokemon.plot.scatter(x='Attack', y='Defense')
pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=20)
#My code

pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=20)
pokemon.Legendary.unique()
#Some univariate plotting here as well :P

pokemon["Legendary"].value_counts().plot.bar()
pokemon.Generation.unique()
#Some univariate plotting here as well :P

pokemon["Generation"].value_counts().plot.bar()
pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean() #Only numerical columns are automatically taken into consideration since mean() is used

print(pokemon_stats_legendary)

print(pokemon_stats_legendary.columns)
pokemon_stats_legendary = pokemon_stats_legendary[['Attack', 'Defense']]

print(pokemon_stats_legendary.columns)
len(pokemon_stats_legendary)
pokemon_stats_legendary #the complete data frame
pokemon_stats_legendary.info()
pokemon_stats_legendary.plot.bar(stacked=True)
#My code

pokemon_stats_legendary.plot.bar(stacked=True)
#My code

pokemon_stats_legendary.plot.area()
#My code

pokemon_stats_legendary.plot.line()
pokemon_stats_by_generation = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
pokemon_stats_by_generation.plot.line()
#My code

pokemon_stats_by_generation.plot.line()