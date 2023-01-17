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
wine = pd.read_csv('../input/winemag-data_first150k.csv')

wine.info()
wine = wine.drop('Unnamed: 0', axis = 1)

wine.head()
wine[['points', 'price']].describe()
import matplotlib.pyplot as plt

import seaborn as sns

fig, axes = plt.subplots(2, 1, figsize = (12, 8))

g = sns.countplot('points', data = wine, ax = axes[0])

g.set_title('Points Distribution', size = 15)



t = sns.distplot(np.log(wine.price.dropna()), ax = axes[1])

t.set_title("Price Distribuition", fontsize = 15)

t.set_xlabel("Log Prices")



plt.tight_layout()

plt.show()
sns.set()

def pop_plot(df, column, num):

    df.groupby(column).size().nlargest(num).plot(kind = 'bar', figsize = (12, 5))

    plt.xticks(rotation = 45)

    plt.xlabel('')

    plt.title('Top Most Reviews' + ' - ' + column, size = 15)

    plt.show()
wine_good = wine[(wine.points >= 90) & (wine.price >= 200)]

wine_better = wine[(wine.points >= 90) & (wine.price < 200)]

pop_plot(wine_good, 'country', 10)

pop_plot(wine_better, 'country', 10)
pop_plot(wine_good, 'variety', 10)

pop_plot(wine_better, 'variety', 10)
wine_country_top4 = wine[wine.country.isin(['France', 'Italy', 'US', 'Germany'])]

g = sns.FacetGrid(wine_country_top4, col = 'country', col_wrap = 2, height = 4, aspect = 3)

g.map(sns.scatterplot, x = wine_country_top4.points, y = wine_country_top4.price)

plt.show()
wine_variety_top4 = wine[wine.variety.isin(['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Bordeaux-style Red Blend'])]

g = sns.FacetGrid(wine_variety_top4, col = 'variety', col_wrap = 2, height = 4, aspect = 3)

g.map(sns.scatterplot, x = wine_variety_top4.points, y = wine_variety_top4.price)

plt.show()
wine_us_france = wine[wine.country.isin(['France', 'US'])]

wine_france = wine[wine.country == 'France']

pop_plot(wine_us, 'province', 5)

pop_plot(wine_france, 'province', 5)
wine_us_ca = wine[wine.province == 'California']

wine_france_bordeaux = wine[wine.province == 'Bordeaux']

pop_plot(wine_us_ca, 'variety', 5)

pop_plot(wine_france_bordeaux, 'variety', 5)
wine_us_france = pd.concat([wine_us_ca, wine_france_bordeaux])

fig, axes = plt.subplots(1, 2, figsize = (15, 5))

sns.boxplot(x = 'country', y = 'points', data = wine_us_france, ax = axes[0]).set_title('Points Comparision between US & France', size = 15)

sns.boxplot(x = 'country', y = 'price', data = wine_us_france, ax = axes[1]).set_title('Price Comparision between US & France', size = 15)

plt.show()
length = wine.description.str.len()

(min(length), max(length))
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

text = ''.join(descript for descript in wine.description)

stopwords = set(STOPWORDS)

wine_sw = ['wine', 'feel', 'note', 'nose', 'show', 'shows', 'touch', 'mouth', 'taste', 'offer']

stopwords.update(wine_sw)

cloud = WordCloud(background_color = 'white', stopwords = stopwords).generate(text)

plt.figure(figsize = (12, 8))

plt.axis('off')

plt.imshow(cloud, interpolation = 'bilinear')

plt.show()
wine_bordeaux_red = wine[wine.variety == 'Bordeaux-style Red Blend']

text_bordeaux_red = ''.join(descript for descript in wine_bordeaux_red.description)

stopwords_bordeaux_red = stopwords.union(['character', 'bordeaux blend', 'good'])

cloud_bordeaux_red = WordCloud(background_color = 'black', stopwords = stopwords_bordeaux_red).generate(text_bordeaux_red)

plt.figure(figsize = (12, 8))

plt.axis('off')

plt.imshow(cloud_bordeaux_red, interpolation = 'bilinear')

plt.show()
wine_pinot = wine[wine.variety == 'Pinot Noir']

text_wine_pinot = ''.join(descript for descript in wine_pinot.description)

stopwords_pinot = stopwords.union(['Pinot Noir', 'character'])

cloud_pinot = WordCloud(background_color = 'black', stopwords = stopwords_pinot).generate(text_wine_pinot)

plt.figure(figsize = (12, 8))

plt.axis('off')

plt.imshow(cloud_pinot, interpolation = 'bilinear')

plt.show()