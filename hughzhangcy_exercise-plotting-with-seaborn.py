import pandas as pd

import seaborn as sns



reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

reviews.head()
sns.countplot(reviews['points'])
sns.kdeplot(reviews.query('price < 200').price)
reviews[reviews['price']<200].price.value_counts().sort_index().plot.line()
sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))
sns.distplot(reviews['points'], bins=10, kde=False)
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex', 

              gridsize=20)
df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]



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
reviews.head()
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

pokemon.head()
sns.countplot(pokemon['Generation'])
sns.distplot(pokemon['HP'])
sns.jointplot(x='Attack', y='Defense', data=pokemon)
sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex')
sns.kdeplot(pokemon['HP'], pokemon['Attack'])
sns.boxplot(x='Legendary', y='Attack', data=pokemon)
sns.violinplot(x='Legendary', y='Attack', data=pokemon)