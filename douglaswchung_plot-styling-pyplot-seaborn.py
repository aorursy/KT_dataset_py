import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.bar(figsize=(12, 6))
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred'
)
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16,
    title='Rankings Given by Wine Magazine',
)
import matplotlib.pyplot as plt

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
import matplotlib.pyplot as plt
import seaborn as sns

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
sns.despine(bottom=True, left=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)
ax = pokemon.plot.scatter(x="Attack", y="Defense", title="Pokemon by Attack and Defense",figsize=(12,6))

ax = pokemon['Total'].plot.hist(bins=40, figsize = (12,6), color="gray", fontsize=16)
ax.set_title("Pokemon by Stat Total", fontsize=20)
ax = pokemon['Type 1'].value_counts().plot.bar(figsize=(12,6), fontsize=16, color="lightblue")
ax.set_title('Pokemon by Primary Type', fontsize=20)
sns.despine(bottom=True, left=True)