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
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
reviews.info()
reviews['points'].value_counts().sort_index().plot.bar(figsize=(12,6))
reviews['points'].value_counts().sort_index().plot.bar(figsize=(12,6),
                                                      color='green')
reviews['points'].value_counts().sort_index().plot.bar(figsize=(12,6),
                                                      color='pink',
                                                       fontsize=16
                                                      )
reviews['points'].value_counts().sort_index().plot.bar(figsize=(12,6),
                                                      color='pink',
                                                       fontsize=16,
                                                       title='Ranking given Wine Magazine'
                                                      )
import matplotlib.pyplot as plt

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='orange',
    fontsize=16
)

ax.set_title("Price of wines given by Wine Magazine", fontsize="20")
import matplotlib.pyplot as plt
import seaborn as sns

ax = reviews['points'].value_counts().sort_index().plot.bar(
     figsize=(12, 6),
     color='mediumvioletred',
     fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
sns.despine(bottom=True, left=True)
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head()
pokemon.plot.scatter(x='Attack', y='Defense', title='Pokemon by Attack and Defense',figsize=(12,6))
pokemon['Total'].plot.hist(
    title='Pokemon by stat total',
    figsize=(12,6),
    color='grey',
    bins=50,
    fontsize=16
) 
pokemon['Type 1'].value_counts().plot.bar(
             figsize=(12,6),
              color='pink',
            fontsize=16,
           title='Pokemon by Primary Type')