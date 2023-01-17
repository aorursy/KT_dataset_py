import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans # KMeans clustering
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
import json
%matplotlib inline 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/tmdb_5000_movies.csv')
movies.head()
credits = pd.read_csv('../input/tmdb_5000_credits.csv')
credits.head()
movies = pd.merge(movies, credits, left_on='id', right_on='movie_id')
movies.head()
movies['genres_num'] = movies['genres'].apply(lambda x: len(json.loads(x)))
movies['keywords_num'] = movies['keywords'].apply(lambda x: len(json.loads(x)))
movies['production_companies_num'] = movies['production_companies'].apply(lambda x: len(json.loads(x)))
movies['production_countries_num'] = movies['production_countries'].apply(lambda x: len(json.loads(x)))
movies['spoken_languages_num'] = movies['spoken_languages'].apply(lambda x: len(json.loads(x)))
movies['cast_num'] = movies['cast'].apply(lambda x: len(json.loads(x)))
movies['crew_num'] = movies['crew'].apply(lambda x: len(json.loads(x)))

str_list = ['id', 'movie_id'] # lista de colunas do tipo string
for colname, colvalue in movies.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# pega todas as colunas numéricas com exceção do id          
num_list = movies.columns.difference(str_list)

movies_num = movies[num_list]
movies_num.head()
# preencher dados inválidos com 0
movies_num = movies_num.fillna(value=0, axis=1)
X = movies_num.values
# normalizar os dados
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
for column in movies_num.columns.difference(['vote_average']):
    movies.plot(y='vote_average', x=column, kind='hexbin', gridsize=45, sharex=False, colormap='cubehelix', title=f'Hexbin de vote_average por {column}', figsize=(12,8))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))
plt.title('Correlação entre os Atributos dos Filmes')
# desenha o mapa de calor
sns.heatmap(movies_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
Nc = range(1, 30)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X_std).score(X_std) for i in range(len(kmeans))]

plt.plot(Nc,score)
plt.xlabel('Número de Agrupamentos')
plt.ylabel('Score')
plt.title('Curva Cotovelo')
plt.show()
# KMeans clustering
kmeans = KMeans(n_clusters=2)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(X_std)

# definindo as cores
LABEL_COLOR_MAP = {0: 'r', 1: 'g'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# criar um DataFrame temporário
df = pd.DataFrame(X_std)
df['X_cluster'] = X_clustered

# Call Seaborn's pairplot to visualize our KMeans clustering
sns.pairplot(df, hue='X_cluster', palette='Dark2', diag_kind='kde', height=3)
# plotar o gráfico de dispersão
plt.figure(figsize = (7,7))
plt.scatter(X_std[:,0],X_std[:,11], c=label_color, alpha=0.5) 
plt.show()
# plotar o gráfico de dispersão
plt.figure(figsize = (7,7))
plt.scatter(X_std[:,0],X_std[:,5], c=label_color, alpha=0.5) 
plt.show()
# KMeans clustering
kmeans = KMeans(n_clusters=3)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(X_std)

# definindo as cores
LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# criar um DataFrame temporário
df = pd.DataFrame(X_std)
df['X_cluster'] = X_clustered

# Call Seaborn's pairplot to visualize our KMeans clustering
sns.pairplot(df, hue='X_cluster', palette='Dark2', diag_kind='kde', height=3)
# plotar o gráfico de dispersão
plt.figure(figsize = (7,7))
plt.scatter(X_std[:,0],X_std[:,9], c=label_color, alpha=0.5) 
plt.show()
columns = list()

for index, value in movies['cast'].iteritems():
    cast = json.loads(value)
    for actor in cast:
        if not (actor['name'] in columns):
            columns.append(actor['name'])

num_movies, _ = movies.shape
data = np.zeros((num_movies, len(columns)))
actors = pd.DataFrame(data, columns=columns)

for index, value in movies['cast'].iteritems():
    cast = json.loads(value)
    for actor in cast:
        actors.loc[index][actor['name']] = 1

actors.head()

actors['ruim'] = movies['vote_average'].apply(lambda x: x <= 2)
actors['regular'] = movies['vote_average'].apply(lambda x: (x > 2 and x <= 5))
actors['bom'] = movies['vote_average'].apply(lambda x: (x > 5 and x <= 7))
actors['otimo'] = movies['vote_average'].apply(lambda x: (x > 7 and x <= 9))
actors['excelente'] = movies['vote_average'].apply(lambda x: x > 9)

actors.head()
frequent_itemsets = apriori(actors, min_support=0.002, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.8)]
columns = list()

for index, value in movies['crew'].iteritems():
    crew = json.loads(value)
    for person in crew:
        if not (person['name'] in columns):
            columns.append(person['name'])

num_movies, _ = movies.shape
data = np.zeros((num_movies, len(columns)))
people = pd.DataFrame(data, columns=columns)

for index, value in movies['crew'].iteritems():
    crew = json.loads(value)
    for person in crew:
        people.loc[index][person['name']] = 1

people.head()
people['ruim'] = movies['vote_average'].apply(lambda x: x <= 2)
people['regular'] = movies['vote_average'].apply(lambda x: (x > 2 and x <= 5))
people['bom'] = movies['vote_average'].apply(lambda x: (x > 5 and x <= 7))
people['otimo'] = movies['vote_average'].apply(lambda x: (x > 7 and x <= 9))
people['excelente'] = movies['vote_average'].apply(lambda x: x > 9)

people.head()
frequent_itemsets = apriori(people, min_support=0.002, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.8)]