import numpy as np 



import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'



import ast
# Importando dados

df_filmes = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)



# Listando as 5 primeiras linhas do arquivo importado

df_filmes.head()
# Exibindo informações do arquivo importado

df_filmes.info()
# Corrige datas de lançamento inexistentes para o ano de 1870 para serem futuramente ignoradas

df_filmes['release_date'] = df_filmes['release_date'].fillna('1870-01-01')



# Criação da coluna Ano de produção

# Utilizando o campo release_date

df_filmes['ano'] = pd.to_datetime(df_filmes['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)



# Transforma a duraçao do filme para 0, quando inexistente

df_filmes['runtime'] = df_filmes['runtime'].fillna('0.0')



# Remove os itens duplicados

df_filmes = df_filmes.drop_duplicates(subset='id', keep="first")
# Cria um dataframe de dimensão pais produtor do filme

# Transofrma o campo production_countries, que está no formato JSON

df_pais = df_filmes[['id', 'production_countries']]

df_pais['production_countries'] = df_pais['production_countries'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])



# Cria um dataframe com o ID do filme e o país produtor

# O ID do filme pode se repetir uma vez que um filme pode ter vários países produtores em conjunto

rows = []

_ = df_pais.apply(lambda row: [rows.append([row['id'], nn]) 

                         for nn in row.production_countries], axis=1)

df_pais = pd.DataFrame(rows, columns=df_pais.columns)
# Cria um dataframe de dimensão gênero do filme

# Transofrma o campo genres, que está no formato JSON

df_genero_filmes = df_filmes[['id', 'genres']]

df_genero_filmes['genres'] = df_genero_filmes['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['id'] for i in x] if isinstance(x, list) else [])



# Cria um dataframe com o ID do filme e o ID do genero

# O ID do filme pode se repetir uma vez que um filme pode ter vários gêneros

rows = []

_ = df_genero_filmes.apply(lambda row: [rows.append([row['id'], nn]) 

                         for nn in row.genres], axis=1)

df_genero_filmes = pd.DataFrame(rows, columns=df_genero_filmes.columns)
# Cria um dataframe de dimensão tipos_genero

df_genero = df_filmes['genres'].dropna().apply(ast.literal_eval)

df_genero = pd.concat([pd.DataFrame(x) for x in df_genero], sort=False)



# Remove os itens duplicados e gera o dataframe final com o ID do genero e sua descrição

df_genero =  df_genero.groupby(by=["id"], as_index=False).first()
# Cria um dataframe de dimensão produtor do filme

# Transofrma o campo production_companies, que está no formato JSON

df_produtor_filmes = df_filmes[['id', 'production_companies']]

df_produtor_filmes['production_companies'] = df_produtor_filmes['production_companies'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['id'] for i in x] if isinstance(x, list) else [])



# Cria um dataframe com o ID do filme e o ID do produtor

# O ID do filme pode se repetir uma vez que um filme pode ter sido produzido por vairos produtores

rows = []

_ = df_produtor_filmes.apply(lambda row: [rows.append([row['id'], nn]) 

                         for nn in row.production_companies], axis=1)

df_produtor_filmes = pd.DataFrame(rows, columns=df_produtor_filmes.columns)
# Cria um dataframe de dimensão produtores

# Cada produtor terá um ID

df_produtor = df_filmes['production_companies'].dropna().apply(ast.literal_eval)

df_produtor =  df_produtor[df_produtor.astype(bool) & df_produtor.notnull()]



df_produtor = pd.concat([pd.DataFrame(x) for x in df_produtor], sort=False)



# Remove os itens duplicados e gera o dataframe final com o ID do genero e sua descrição

df_produtor =  df_produtor.groupby(by=["id"], as_index=False).first()
# Limpeza de colunas desnecessárias

df_filmes = df_filmes.drop(['imdb_id', 'homepage', 'original_title', 'production_companies', 'genres', 'production_countries', 'belongs_to_collection', 'video', 'popularity', 'adult', 'spoken_languages'], axis=1)



# Limpeza do texto da descrição

df_filmes['overview'] = df_filmes['overview'].str.replace(';',',').str.replace('\n','', regex=True)



# Limpeza do texto do resumo

df_filmes['tagline'] = df_filmes['tagline'].str.replace(';',',').str.replace('\n','', regex=True)
# Exporta DataFrames para arquivos CSV



df_pais.to_csv('df_pais.csv', sep=';', index=False)

df_genero_filmes.to_csv('df_genero_filmes.csv', sep=';', index=False)

df_genero.to_csv('df_genero.csv', sep=';', index=False)

df_produtor_filmes.to_csv('df_produtor_filmes.csv', sep=';', index=False)

df_produtor.to_csv('df_produtor.csv', sep=';', index=False)

df_filmes.to_csv('df_filmes.csv', sep=';', index=False)
# Gera link CSV

#from IPython.display import HTML

#def create_download_link(filename):  

#    title = "Download CSV file" 

#    html = '<a href={filename}>{title}</a>'

#    html = html.format(title=title,filename=filename)

#    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

#create_download_link('df_filmes.csv')