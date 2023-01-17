# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# for visualizations

import matplotlib.pyplot as plt

import squarify

import seaborn as sns

plt.style.use('fivethirtyeight')



# for analysis

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules       

        

        

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/movielens/movies.csv')



# let's check the shape of the dataset

data.shape
#verifico los nulos 

data.info()
#Head of the data

data.head(10)
# checkng the tail of the data



data.tail()
# checking the random entries in the data



data.sample(10)
# let's describe the dataset



data.describe()
movies_genero = data.drop('genres',1).join(data.genres.str.get_dummies())

pd.options.display.max_columns=100
#listo las peliculas y los distintos generos que tiene

movies_genero.head()
stat1 = movies_genero.drop(['title', 'movieId'],1).apply(pd.value_counts)

stat1.head()
stat1 = stat1.transpose().drop(0,1).sort_values(by=1, 

                                                ascending=False).rename(columns={1:'No. of movies'})

#coloco la ´primera columna con el nombre genre

stat1.index = stat1.index.set_names('genre')
#obtengo la cantidad de peliculas por genero, ordenadas de mayor a menor.

stat1.head()
stat1.tail()
#genero una nueva tabla con la columna numero de generos que tiene una pelicula.



stat2 = data.join(data.genres.str.split('|').reset_index().genres.str.len(), rsuffix='r').rename(

    columns={'genresr':'genre_count'})
stat2.head(10)
stat2.tail(10)
#agrupo por generos y cuento cuantas peliculas hay por genero.

stat2 = stat2[stat2['genre_count']==1].drop('movieId',1).groupby('genres').sum().sort_values(

    by='genre_count', ascending=False)
stat2.head(10)
stat2.tail(10)
#obtengo el total de distintos generos.

stat2.shape
#uniendo la peliculas por genero, y visualizando cuantas peliculas tienen asociado un unico genero.

stat = stat1.merge(stat2, how='left', left_index=True, right_index=True).fillna(0)
stat.genre_count=stat.genre_count.astype(int)

stat.rename(columns={'genre_count': 'No. of movies with only 1 genre'},inplace=True)
stat.head()
stat.tail()
import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud



plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data['genres']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Items',fontsize = 20)

plt.show()
# looking at the frequency of most popular items 



plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))

data['genres'].value_counts().head(40).plot.bar(color = color)

plt.title('frequency of most popular items', fontsize = 20)

plt.xticks(rotation = 90 )

plt.grid()

plt.show()
y = data['genres'].value_counts().head(50).to_frame()

y.index
# plotting a tree map



plt.rcParams['figure.figsize'] = (20, 20)

color = plt.cm.cool(np.linspace(0, 1, 50))

squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)

plt.title('Tree Map for Popular Items')

plt.axis('off')

plt.show()
#movies_genero

#visualizo las pprimeras 15 peliculas por genero aventura



movies_genero['Adventure'] = 'adventure'

food = movies_genero.truncate(before = -1, after = 15)
import networkx as nx



food = nx.from_pandas_edgelist(food, source = 'Adventure', target = 'title', edge_attr = True)
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

pos = nx.spring_layout(food)

color = plt.cm.Wistia(np.linspace(0, 15, 1))

nx.draw_networkx_nodes(food, pos, node_size = 15000, node_color = color)

nx.draw_networkx_edges(food, pos, width = 3, alpha = 0.6, edge_color = 'black')

nx.draw_networkx_labels(food, pos, font_size = 20, font_family = 'sans-serif')

plt.axis('off')

plt.grid()

plt.title('Top 15 First Choices', fontsize = 40)

plt.show()
#movies_genero

#visualizo las pprimeras 15 peliculas por genero aventura

movies_genero['Comedy'] = 'Comedy'

food = movies_genero.truncate(before = -1, after = 15)
import networkx as nx



food = nx.from_pandas_edgelist(food, source = 'Comedy', target = 'title', edge_attr = True)
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (20, 20)

pos = nx.spring_layout(food)

color = plt.cm.Wistia(np.linspace(0, 15, 1))

nx.draw_networkx_nodes(food, pos, node_size = 15000, node_color = color)

nx.draw_networkx_edges(food, pos, width = 3, alpha = 0.6, edge_color = 'black')

nx.draw_networkx_labels(food, pos, font_size = 20, font_family = 'sans-serif')

plt.axis('off')

plt.grid()

plt.title('Top 15 First Choices', fontsize = 40)

plt.show()
movies_genero.set_index(['movieId','title'],inplace=True)
movies_genero = data.drop(['movieId','genres','title'],1).join(data.genres.str.get_dummies())

pd.options.display.max_columns=100

movies_genero.head()
frequent_itemsets = apriori(movies_genero, min_support = 0.02, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
# getting th item sets with length = 2 and support more han 4%

frequent_itemsets[ (frequent_itemsets['length'] == 2) &

                   (frequent_itemsets['support'] >= 0.04) ]
# getting th item sets with length = 1 and support more han 5%



frequent_itemsets[ (frequent_itemsets['length'] == 1) &

                   (frequent_itemsets['support'] >= 0.05) ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Comedy', 'Drama'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Adventure', 'Action'} ]
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Comedy', 'Romance'} ]