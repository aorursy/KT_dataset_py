import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

import seaborn as sns
reviews.head()
sns.countplot(reviews['points'])
sns.kdeplot(reviews.query('price < 200').price)
#Or

sns.kdeplot(reviews[reviews['price'] < 200]['price'])
reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()
#My code to plot relative proportions of different prices

(reviews[reviews['price'] < 200]['price'].value_counts().sort_index()/len(reviews)).plot.line()
ax = sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))
#Alternatively

nan_df = reviews[reviews['price'] < 200][['price', 'points']]

print(nan_df.head())

print("********************************************************************************")

import numpy as np

print(np.where(pd.isnull(nan_df)))

print("So no NaN's")

print("********************************************************************************")

sns.kdeplot(nan_df.sample(5000))
#Alternatively

sns.kdeplot(reviews[reviews['price'] < 200][['price', 'points']].dropna().sample(5000))
sns.distplot(reviews['points'], bins=10, kde=False)
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex', 

              gridsize=20)
#My code to verify the histogram on the sides of the above jointplot(hex/scatter)

import matplotlib.pyplot as plt

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(26,6))

sns.distplot(reviews['points'], ax=axarr[0], kde=False).set_title('points', fontsize=20)

sns.distplot(reviews[reviews['price'] < 100]['price'], ax=axarr[1], kde=False).set_title('price', fontsize=20)
reviews['variety'].value_counts().head(100).plot.bar(figsize=(30,15), fontsize=15).set_title("Wine variety", fontsize=30)
#Mycode for considering 15 Variety values with highest value_counts()

fig, axarr = plt.subplots(1,1,figsize=(30,15))

sns.boxplot(x='variety', y='points', data=reviews[reviews.variety.isin(reviews.variety.value_counts().head(15).index)], ax=axarr).set_title("Box Plot of Variety vs Points", fontsize=30)
#Considering only 5 wine varieties that have the highest counts

df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

print(df.variety.unique())

df.head()
df['variety'].value_counts().head(100).plot.bar(figsize=(30,15), fontsize=15).set_title("Top 5 Wine variety by count", fontsize=30)
sns.boxplot(

    x='variety',

    y='points',

    data=df

)
sns.violinplot(

    x='variety',

    y='points',

    data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]

)
sns.violinplot(x='variety', y='price', data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)])
reviews.head()
from IPython.display import HTML

HTML("""

<ol>

<li>A seaborn countplot is like a pandas bar plot.</li>

<li>A seaborn jointplot is like a pandas hex plot.</li>

<li>KDEPlots work by aggregating data into a smooth curve. This is great for interval data but doesn't always work quite as well for ordinal categorical data.</li>

<li>The top of the box is the 75th percentile. The bottom of the box is the 25th percentile. The median, the 50th percentile, is the line in the center of the box. So 50% of the data in the distribution is located within the box!</li>

</ol>

""")
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

pokemon.head()
sns.countplot(pokemon['Generation'])
#My code

#pokemon['Generation'].value_counts().sort_index().plot.bar() #similar plot

sns.countplot(pokemon['Generation'])
sns.distplot(pokemon['HP'])
#My code

fig, axarr = plt.subplots(1, 2, figsize=(14,4))

sns.distplot(pokemon.HP, ax=axarr[0]).set_title("Histogram with KDE")

sns.kdeplot(pokemon.HP, ax=axarr[1]).set_title("Just KDE")
sns.jointplot(x='Attack', y='Defense', data=pokemon)
#My code

sns.jointplot(x='Attack', y='Defense', data=pokemon)
sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex')
#My code

sns.jointplot(x='Attack', y='Defense', data=pokemon[['Attack', 'Defense']], kind='hex')
sns.kdeplot(pokemon['HP'], pokemon['Attack'])
#My code

sns.kdeplot(pokemon.HP, pokemon.Attack).set_title("KDE 2D plot")
#My code : Similar plot to above plot

sns.jointplot(x=pokemon.HP, y=pokemon.Attack, data=pokemon)
sns.boxplot(x='Legendary', y='Attack', data=pokemon)
#My code

sns.boxplot(x='Legendary', y='Attack', data=pokemon)
sns.violinplot(x='Legendary', y='Attack', data=pokemon)
#My code

sns.violinplot(x='Legendary', y='Attack', data=pokemon)