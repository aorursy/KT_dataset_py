import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.bar(figsize=(20, 12))
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(20, 12),
    color='mediumvioletred'
)
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(20,12),
    color='mediumvioletred',
    fontsize=18
)
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(20,12),
    color='mediumvioletred',
    fontsize=16,
    title='Wine Magazine Rankings',
)
import matplotlib.pyplot as plt

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(20,12),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Wine Magazine Rankings", fontsize=25)
import matplotlib.pyplot as plt
import seaborn as sns

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(20,12),
    color='mediumvioletred',
    fontsize=18
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
sns.despine(bottom=True, left=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)
pokemon.plot.scatter(x='Attack', y='Defense',
                     figsize=( 20,12),
                     title='Pokemon by Attack and Defense',color='blue',fontsize=18)
#pokemon.plot.area(x='Attack', y='Defense',
                   #  figsize=( 20,12),
                    # title='Pokemon by Attack and Defense',color='blue',fontsize=18)
ax = pokemon['Total'].plot.hist(
    figsize=(20,12),
    fontsize=14,
    bins=50,
    color='gray'
)
ax.set_title('Pokemon by Stat Total', fontsize=18)
ax = pokemon['Type 2'].value_counts().plot.bar(
    figsize=(20,12),
    fontsize=18,color='red'
)
ax.set_title("Pokemon by Primary Type", fontsize=18)
sns.despine(bottom=True, left=True)