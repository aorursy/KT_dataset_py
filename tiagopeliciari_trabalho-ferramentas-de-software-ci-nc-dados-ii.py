# Importação das bibliotecas genéricas mais usadas

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import ast

from wordcloud import WordCloud, STOPWORDS

from IPython.display import Image, HTML
# Importando dados

df_filmes = pd.read_csv('../input/movies_metadata.csv', low_memory=False)



# Listando as 5 primeiras linhas do arquivo importado

df_filmes.head()
# Exibindo informações do arquivo importado

df_filmes.info()
# Limpeza de colunas desnecessárias

df_filmes = df_filmes.drop(['imdb_id', 'homepage', 'original_title'], axis=1)



# Criação da coluna ano para utilização posterior

# Utilizando o campo release_date

df_filmes['ano'] = pd.to_datetime(df_filmes['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
df_filmes['runtime'].describe()
# Converão para float

df_filmes['runtime'] = df_filmes['runtime'].astype('float')



# Plotando o gráfico

plt.figure(figsize=(12,6))

sns.distplot(df_filmes[(df_filmes['runtime'] < 300) & (df_filmes['runtime'] > 0)]['runtime'])

plt.xlabel('Duração em minutos')

plt.ylabel("Quantidade de filmes")

plt.title("Distribuição da duraçao dos filmes")

# Tratamento de Dado

# Transformação da coluna production_countries que é um json, para extrair o país produtos

df_filmes['production_countries'] = df_filmes['production_countries'].fillna('[]').apply(ast.literal_eval)

df_filmes['production_countries'] = df_filmes['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])



s = df_filmes.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'countries'



# Criação de novo Data Set para armazenar a quantidade de filmes por país, agrupando e contando.

con_df = df_filmes.drop('production_countries', axis=1).join(s)

con_df = pd.DataFrame(con_df['countries'].value_counts())

con_df['country'] = con_df.index

con_df.columns = ['Quantidade', 'Pais']

con_df = con_df.reset_index().drop('index', axis=1)



# Listando o TOP 20

con_df.head(20)
# Criação do plot horizontal

sns.set(style="whitegrid")

plt.subplots(figsize=(8, 15))

sns.set_color_codes("pastel")

sns.barplot(x="Quantidade", y="Pais", data=con_df.head(20),

            label="Total", color="b")

plt.xlabel('Quantidade de filmes produzidos')

plt.ylabel("País")

plt.title("TOP 20 filmes por país produtor")
# Junta todos os nomes em inglês para a variável todos_titulos

df_filmes['title'] = df_filmes['title'].astype('str')

todos_titulos = ' '.join(df_filmes['title'])



# Plota o wordcloud

title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(todos_titulos)

plt.figure(figsize=(16,8))

plt.imshow(title_wordcloud)

plt.axis('off')

plt.show()
# Dropa as linguagens duplicadas 

df_filmes['original_language'].drop_duplicates().shape[0]
# Agrupa por idioma (original_language) e conta a quantidade de filmes por idioma

idioma = pd.DataFrame(df_filmes['original_language'].value_counts())

idioma['language'] = idioma.index

idioma.columns = ['Quantidade', 'Idioma']



# Lista os 20 idiomas mais frequentes

idioma.head(20)
# Transformação do gênero uma vez que o armazenado é um JSON

df_filmes['genres'] = df_filmes['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = df_filmes.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
# Criar o generos_df para contagem e classificação

generos_df = df_filmes.drop('genres', axis=1).join(s)

generos_df['genre'].value_counts().shape[0]
# Pega os 20 principais gêneros e conta quantos filmes possuem nesses gêneros

generos_populares = pd.DataFrame(generos_df['genre'].value_counts()).reset_index()

generos_populares.columns = ['Genero', 'Filmes']



# Lista os 20 gêneros mais frequentes

generos_populares.head(20)
# Plota filmes por gênero num barplot vertical

plt.figure(figsize=(28,10))

sns.barplot(x='Genero', y='Filmes', data=generos_populares.head(20))

plt.xlabel("Gênero")

plt.ylabel("Quantidade")

plt.title("Quantidade de filmes por gênero - TOP 20")

plt.show()
# Arrumando o endereço da imagem do poster, acrescentando os parêmtros HTML

base_poster_url = 'http://image.tmdb.org/t/p/w185/'

df_filmes['poster_path'] = "<img src='" + base_poster_url + df_filmes['poster_path'] + "' style='height:100px;'>"



# Gera a lista de filmes, ordenando por maior receita e as 10 maiores arrecadaçoes

maiores_receitas = df_filmes[['poster_path', 'title', 'revenue', 'ano']].sort_values('revenue', ascending=False).head(10)



# Transforma em bilhões de dolares a escala para ficar mais claro a análise

maiores_receitas['revenue'] = maiores_receitas['revenue'] / 1000000000

maiores_receitas.columns = ['Poster', 'Titulo', 'Receitas (bilhões)', 'Ano']

pd.set_option('display.max_colwidth', 100)



# Exibe o poster utilizando o endereço da imagem armazenada no dataset

HTML(maiores_receitas.to_html(escape=False))
# Gera a lista de filmes, ordenando por maior receita

maiores_receitas_ptbr = df_filmes[['poster_path', 'title', 'revenue', 'ano', 'original_language']]



# Filtra os filmes com idioma português e as 20 maiores arrecadaçoes

maiores_receitas_ptbr = maiores_receitas_ptbr[maiores_receitas_ptbr['original_language'] == 'pt' ].sort_values('revenue', ascending=False).head(20)



# Transforma em milhões de dolares a escala para ficar mais claro a análise

maiores_receitas_ptbr['revenue'] = maiores_receitas_ptbr['revenue'] / 1000000

maiores_receitas_ptbr.columns = ['Poster', 'Titulo', 'Receita (milhões dolares)', 'Ano', 'Idioma']

pd.set_option('display.max_colwidth', 100)



# Exibe o poster utilizando o endereço da imagem armazenada no dataset

HTML(maiores_receitas_ptbr.to_html(escape=False))