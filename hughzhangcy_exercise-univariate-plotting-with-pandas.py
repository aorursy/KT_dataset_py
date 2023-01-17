import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head(3)
reviews['province'].value_counts().head(10).plot.bar()
(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()
reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.line()
reviews['points'].value_counts().sort_index().plot.area()
reviews[reviews['price'] < 200]['price'].plot.hist()
reviews['price'].plot.hist()
reviews[reviews['price'] > 1500]
reviews['points'].plot.hist()
pd.set_option('max_columns', None)

pokemon = pd.read_csv("../input/pokemon/pokemon.csv")

pokemon.head(3)
pokemon['type1'].value_counts().plot.bar()
pokemon['hp'].value_counts().sort_index().plot.line()
pokemon['weight_kg'].plot.hist()