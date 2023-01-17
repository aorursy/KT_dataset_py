import pandas as pd



notas = pd.read_csv('../input/ratingscsv/ratings.csv')
notas.head() #lista as 5 primeiras linhas
notas.shape #formato (linhas e colunas)
notas.columns = ['usuarioId', 'filmeId', 'nota', 'momento'] #alterar o nome das colunas
notas.head()    #Documentação PANDAS DATAFRAME
notas['nota']   #para listas utilizar []     # PANDAS SERIES
notas['nota'].unique()   #valores distintos
notas['nota'].value_counts()   #conta valores distintos ordenados pro frequência
notas['nota'].mean()    #media
notas.nota   #seleção de coluna
notas.nota.head()
notas.nota.plot()
notas.nota.plot(kind='hist')   #histograma da coluna notas
print('Media =',notas['nota'].mean())

print('Mediana =',notas['nota'].median())
notas.nota.describe()    #descrição dos dados   
import seaborn as sns   #SEABORN outra biblioteca de visualização
sns.boxplot(notas.nota)
filmes = pd.read_csv('../input/movies/movies.csv')
filmes.columns = ['filmeId','titulo','genero']

filmes.head()
notas.head()
notas.query('filmeId==1').nota.mean()  #Analisando algumas notas especificas por filme
notas.query('filmeId==2').nota.mean()
medias_por_filme = notas.groupby('filmeId').mean().nota  #GroupBy com resultado em média

medias_por_filme.head()
medias_por_filme.plot(kind='hist')
sns.boxplot(medias_por_filme)
sns.boxplot(y=medias_por_filme)
medias_por_filme.describe()
sns.distplot(medias_por_filme, bins=10)   #histograma no SEABORN

plt.title('Histograma das medias dos filmes')
import matplotlib.pyplot as plt
plt.hist(medias_por_filme)

plt.title('Histograma das medias dos filmes')
sns.boxplot(y=medias_por_filme)

plt.figure(figsize=(10,100))
tmdb = pd.read_csv('../input/tmdb-5000-movies/tmdb_5000_movies.csv')

tmdb.head()
tmdb.original_language.unique() #categoria nominal
# primeiro grau #categoria ordinal

# segundo grau

# terceiro grau
# budget => orçamento => quantitativa contínuo
#quantidade de votos => 1,2,3,4 sempre inteiro. quantitativa intervalar
tmdb['original_language'].value_counts().index
tmdb['original_language'].value_counts().values
contagem_de_lingua = tmdb['original_language'].value_counts().to_frame().reset_index()  #série

contagem_de_lingua.columns = ['original_language','total']

contagem_de_lingua.head()
# para buscar diversas formas de plotar categorias no seaborn categorical plot
tmdb['original_language'].value_counts().values
tmdb.original_language.value_counts()
sns.barplot(x='original_language', y='total',data = contagem_de_lingua)
sns.catplot(x='original_language' , kind='count', data = tmdb)
# !pip install seaborn
plt.pie(contagem_de_lingua['total'],labels = contagem_de_lingua['original_language'])
total_por_lingua = tmdb["original_language"].value_counts()

total_por_lingua.loc["en"]
total_por_lingua = tmdb["original_language"].value_counts()

total_geral = total_por_lingua.sum()

total_de_ingles = total_por_lingua.loc["en"]

total_do_resto = total_geral - total_de_ingles

print(total_de_ingles, total_do_resto)
dados = {

    'lingua' : ['ingles','outros'],

    'total' : [total_de_ingles, total_do_resto]



}



dados = pd.DataFrame(dados)

dados
sns.barplot(data = dados, x = 'lingua', y = 'total')
plt.pie(dados.total, labels = dados.lingua)
total_por_lingua_de_outros_filmes = tmdb.query("original_language != 'en'").original_language.value_counts()

total_por_lingua_de_outros_filmes
filmes_sem_lingua_original_ingles = tmdb.query("original_language != 'en'")

#plt.figure(figsize=(5,10)) o catplot atua em alto nível, não aceitando alteração de tamanho por este tipo de comando

sns.catplot(x='original_language', 

            kind='count', 

            data = filmes_sem_lingua_original_ingles)
sns.catplot(x='original_language', 

            kind='count', 

            data = filmes_sem_lingua_original_ingles, 

            aspect = 2, 

            order = total_por_lingua_de_outros_filmes.index,

           palette = 'GnBu_d')
filmes.head()
notas.head()
notas_do_toy_story = notas.query("filmeId==1")

notas_do_jumanji = notas.query("filmeId==2")

print(len(notas_do_toy_story),len(notas_do_jumanji))
print("nota média do Toy Story %.2f" % notas_do_toy_story.nota.mean())

print("nota média do Jumanji %.2f" % notas_do_jumanji.nota.mean())
print("nota mediana do Toy Story %.2f" % notas_do_toy_story.nota.median())

print("nota mediana do Jumanji %.2f" % notas_do_jumanji.nota.median())
#numpy utilizado para análise numéricas no pandas     abreviado como np

import numpy as np
[2.5] * 10


filme1 = np.append(np.array([2.5] * 10),np.array([3.5] * 10))



filme2 = np.append(np.array([5] * 10),np.array([1] * 10))
print(filme1.mean(),filme2.mean())

print(np.std(filme1),np.std(filme2))

print(np.median(filme1),np.median(filme2))
sns.distplot(filme1)

sns.distplot(filme2)
plt.hist(filme1)

plt.hist(filme2)
sns.boxplot(filme1)

sns.boxplot(filme2)
plt.boxplot([filme1,filme2])
plt.boxplot([notas_do_toy_story.nota,notas_do_jumanji.nota])
sns.boxplot(notas_do_toy_story.nota)

sns.boxplot(notas_do_jumanji.nota)
sns.boxplot(data = notas.query("filmeId in [1,2,3,4,5]"), x = 'filmeId', y = 'nota')
print(notas_do_toy_story.nota.std(),notas_do_jumanji.nota.std())