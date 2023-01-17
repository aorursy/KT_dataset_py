import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/vgsales.csv')
most_popular_genre = (data.groupby('Platform')

                          .apply(lambda x: x['Genre'].value_counts().idxmax())

                          .sort_values())

print(most_popular_genre)
besetsellers_global = []

besetsellers_by_genre = []

for platform, genre in most_popular_genre.iteritems():

    besetsellers_global.append(data.iloc[data[data['Platform'] == platform]['Global_Sales'].idxmax()][['Name', 'Global_Sales', 'Genre']])

    besetsellers_by_genre.append(data.iloc[data[(data['Platform'] == platform) & (data['Genre'] == genre)]['Global_Sales'].idxmax()][['Name', 'Global_Sales', 'Genre']])

besetsellers_global = pd.DataFrame(besetsellers_global, index=most_popular_genre.index)

besetsellers_by_genre = pd.DataFrame(besetsellers_by_genre, index=most_popular_genre.index)

bestsellers = besetsellers_global.join(besetsellers_by_genre, lsuffix='_global', rsuffix='_genre')
ax = bestsellers[['Global_Sales_global', 'Global_Sales_genre']].plot(kind='barh', 

                                                                    figsize=(5, 16),

                                                                    width=0.9)

ax.legend(['Global sales leader', 'Genre sales leader'])

ax.set_yticklabels(list(bestsellers.apply(lambda x: '(' + x['Genre_global'] + ')    ' + x['Name_global']

                                             if x['Name_global'] == x['Name_genre']

                                             else '(' + x['Genre_global'] + ')    ' + x['Name_global'] +

                                             '\n(' + x['Genre_genre'] + ')    ' + x['Name_genre'],

                                    axis=1)), fontsize=9)

y = -0.15

for ix in bestsellers.index:

    ax.text(40, y, ix)

    y += 1

plt.ylabel('Games')

plt.xlabel('Global sales')