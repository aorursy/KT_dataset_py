import matplotlib.pyplot as plt # Bibilioteca util para criar gráficos

import pandas as pd # Bibilioteca para auxiliar a importar e maniular nossos dataframes

from sklearn.tree import DecisionTreeClassifier #responsável pela geração do modelo 

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

import numpy as np # Bibilioteca útil para realizar operações matemáticas

import seaborn as sns # Bibilioteca utilizada para dar um toque especial nos gráficos

#import chardet   #Trabalha com leitura de arquivos, acredito que n será necessário utiliza=lá

plt.style.use('ggplot') #Customização de gráficos

plt.style.use("seaborn-white")

import os

from mpl_toolkits.mplot3d import Axes3D



import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movie.csv')

ratings = pd.read_csv('../input/rating.csv')



movies.sort_values(by='movieId', inplace=True)

movies.reset_index(inplace=True, drop=True)

ratings.sort_values(by='movieId', inplace=True)

ratings.reset_index(inplace=True, drop=True)
#Dimensão  datasets

print("Dimensão  dataset de movies")

print("Colunas:", movies.shape[1],"\nLinhas:", movies.shape[0])

print("-")

print("Dimensão  dataset de ratings")

print("Colunas:", ratings.shape[1],"\nLinhas:", ratings.shape[0])
#Verificando os primeiros registros do conjunto de dados

movies.head()
ratings.head()
ratings.dtypes
movies.dtypes
ratings.info()
ratings.skew()
movies.skew()
# Dividindo o título e o ano de lançamento em colunas separadas no dataframe de filmes

#Convertertendo ano para timestamp.



movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)

movies.year = pd.to_datetime(movies.year, format='%Y')

movies.year = movies.year.dt.year

movies.title = movies.title.str[:-7]
#f,ax = plt.subplots(figsize=(10,8))

#sns.heatmap(ratings.corr(), annot=True, linewidths=.7, fmt= '.2f',ax=ax)

#plt.show()
#categorizando os gêneros de filmes corretamente. 

#Trabalhar mais tarde com + 20MM de linhas de strings consome muito recurso

genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()

genres_unique = pd.DataFrame(genres_unique, columns=['genre']) 

movies = movies.join(movies.genres.str.get_dummies().astype(bool))

movies.drop('genres', inplace=True, axis=1)
genres_unique
movies.head()
# Modificando o formato do registro de data e hora da avaliação

ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)

ratings.timestamp = ratings.timestamp.dt.year
ratings['timestamp'].unique()
sns.heatmap(movies.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
#Contando valores nulos

movies.isnull().sum().sort_values(ascending=False).head(10)
sns.heatmap(ratings.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
#contando valores nulos

ratings.isnull().sum().sort_values(ascending=False).head(10)
# Removendo valores nulos do datasets movies

movies.dropna(inplace=True)
movies.isnull().sum().sort_values(ascending=False).head(10)
#df_mv_year = movies.groupby('movieId')['year']
dftmp = movies[['movieId', 'year']].groupby('year')



fig, ax1 = plt.subplots(figsize=(15,8))

ax1.plot(dftmp.year.first(), dftmp.movieId.nunique(), "b")

ax1.grid(False)



dftmp = ratings[['rating', 'timestamp']].groupby('timestamp')

ax2 = ax1.twinx() #Plotando os dados de avaliações no eixo

ax2.plot(dftmp.timestamp.first(), dftmp.rating.count(), "r")

ax2.grid(False)



ax1.set_xlabel('Ano')

ax1.set_ylabel('Número de filmes liberados'); ax2.set_ylabel('Número de avaliações')

plt.title('Filmes por ano')

plt.show()
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
#Distribuição dos dados de avaliação e ano

plotPerColumnDistribution(ratings, 10, 5)
ratings.columns
#Quantidade de usuários

ratings['userId'].count()
ratings.groupby('timestamp')['userId'].count()
#Quantidade de usuários registrados por ano

ratings.groupby('timestamp')['userId'].count().plot(figsize=(15,8), color="g")

plt.ylabel("Qtd de usuários")

plt.xlabel("Ano")

plt.title("Contagem de usuários por ano")

plt.show()
plt.figure(figsize=(10,5))

dftmp = movies[['movieId', 'year']].groupby('year')

df = pd.DataFrame({'All_movies' : dftmp.movieId.nunique().cumsum()})



#Histograma para cada gênero individual

for genre in genres_unique.genre:

    dftmp = movies[movies[genre]][['movieId', 'year']].groupby('year')

    df[genre]=dftmp.movieId.nunique().cumsum()

df.fillna(method='ffill', inplace=True)

df.loc[:,df.columns!='All_movies'].plot.area(stacked=True, figsize=(15,8))



# Histograma de plotagem para todos os filmes

plt.plot(df['All_movies'], marker='o', markerfacecolor='black')

plt.xlabel('Ano')

plt.ylabel('Acumulativo de filmes por gênero')

plt.title('Total de filmes por gênero') # Many movies have multiple genres, so counthere is higher than number of movies

plt.legend(loc=(1.05,0), ncol=2)

plt.show()



#  dispersão simples do número de filmes marcados com cada gênero

plt.figure(figsize=(15,8))

barlist = df.iloc[-1].sort_values().plot.bar()

barlist.patches[0].set_color('b')

plt.xticks(rotation='vertical')

plt.title('Filmes por gênero')

plt.xlabel('Gênero')

plt.ylabel('Número de filmes')

plt.show()
dftmp = ratings[['movieId','rating']].groupby('movieId').mean()



# inicializando uma lista vazia para capturar estatísticas básicas por gênero

rating_stats = []

# Histograma geral do lote de todas as classificações

dftmp.hist(bins=25, grid=False, edgecolor='b', normed=True, label ='All genres', figsize=(15,8))



# Histograma com linhas kde para melhor visibilidade por gênero

for genre in genres_unique.genre:

    dftmp = movies[movies[genre]==True]

    dftmp = ratings[ratings.set_index('movieId').index.isin(dftmp.set_index('movieId').index)]

    dftmp = dftmp[['movieId','rating']].groupby('movieId').mean()

    dftmp.rating.plot(grid=False, alpha=0.6, kind='kde', label=genre)

    avg = dftmp.rating.mean()

    std = dftmp.rating.std()

    rating_stats.append((genre, avg, std))

plt.legend(loc=(1.05,0), ncol=2)

plt.xlim(0,5)

plt.xlabel('Avaliações')

plt.ylabel('Densidade')

plt.title('Histograma de avaliação de filmes')

plt.show()