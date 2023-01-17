import pandas as pd

import matplotlib.pyplot as plt
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head(3)
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))



reviews['points'].value_counts().sort_index().plot.bar(

    ax=axarr[0]

)



reviews['province'].value_counts().head(20).plot.bar(

    ax=axarr[1]

)
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))



reviews['points'].value_counts().sort_index().plot.bar(

    ax=axarr[0][0]

)



reviews['province'].value_counts().head(20).plot.bar(

    ax=axarr[1][1]

)
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))



reviews['points'].value_counts().sort_index().plot.bar(

    ax=axarr[0][0], fontsize=12, color='mediumvioletred'

)

axarr[0][0].set_title("Wine Scores", fontsize=18)



reviews['variety'].value_counts().head(20).plot.bar(

    ax=axarr[1][0], fontsize=12, color='mediumvioletred'

)

axarr[1][0].set_title("Wine Varieties", fontsize=18)



reviews['province'].value_counts().head(20).plot.bar(

    ax=axarr[1][1], fontsize=12, color='mediumvioletred'

)

axarr[1][1].set_title("Wine Origins", fontsize=18)



reviews['price'].value_counts().plot.hist(

    ax=axarr[0][1], fontsize=12, color='mediumvioletred'

)

axarr[0][1].set_title("Wine Prices", fontsize=18)



plt.subplots_adjust(hspace=.3)



import seaborn as sns

sns.despine()
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")

pokemon.head(3)
fig, axarr = plt.subplots(2, 1, figsize=(8, 8))
fig, axarr = plt.subplots(2, 1, figsize=(8, 8))



pokemon['Attack'].plot.hist(

    ax=axarr[0],

    title='Pokemon Attack Ratings'

)



pokemon['Defense'].plot.hist(

    ax=axarr[1],

    title='Pokemon Defense Ratings'

)