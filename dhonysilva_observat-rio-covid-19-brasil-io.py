# Importa as libs e conecta com o json



import requests

import pandas as pd

from pandas.io.json import json_normalize



# A API tem paginação a cada 1.000 registros, portando é necessário percorrer a paginação.



incremento = []



url = 'https://brasil.io/api/dataset/covid19/caso/data?format=json'



while url:

    response = requests.get(url)

    data = response.json()

    url = data['next']

    incremento += data['results']



# Normaliza os dados  

dados = json_normalize(incremento)
dados.shape
dados.head(10)
dados.dtypes
dados = dados.set_index('date')
dados.head()
dados.index
# Utilização da função loc para filtrar

# cidades diferentes de Nulo

# Permanece o datafame dados com todas as datas.



dados = dados.loc[dados.city.notnull()]
# Utilização da função loc para filtrar

# is_last for Verdadeiro.



dados_cidades = dados.loc[dados.is_last == True]
dados_cidades
dados_cidades.sort_values(by=['confirmed'], ascending=False)
# Armazena a última data



# Com a utilização do is_last = True essa variável não está sendo utilizada agora. Continuo mantendo no documento para uma posterior utilização.



max_date = dados_cidades.index.max()
# Somamos o total de casos confirmados.



dados_cidades.confirmed.sum()
# Somamos o total de óbitos.



dados_cidades.deaths.sum()
por_estado = dados_cidades.groupby(['state'])['confirmed', 'deaths'].max().sort_values(by=['confirmed'], ascending=False)
por_estado
# Optamos por listar as 10 cidades com maior número de casos.



por_cidade_confirmados = dados_cidades.groupby(['city'])['confirmed', 'deaths'].max().sort_values(by=['confirmed'], ascending=False).head(10)
# Para a listagem de óbitos, estamos listados todas as cidades com ocorrências.



por_cidade_obitos = dados_cidades.where(dados_cidades.deaths > 0).groupby(['city'])['confirmed', 'deaths'].max().sort_values(by=['deaths'], ascending=False)
por_cidade_obitos
# Utilização do dataframe dados que contêm todas as datas.



por_data = dados.groupby(['date'])['confirmed', 'deaths'].sum().reset_index()
por_data.sort_values(by=['date'], ascending=False)
import matplotlib.pyplot as plt

# Display figures inline in Jupyter notebook



import matplotlib.dates as mdates
# Dados Confirmados por Estado.



ax = por_estado['confirmed'].plot.bar(color='C0')

ax.set_ylabel('Casos')

#ax.set_ylim(0, 0.3)

ax.set_title('Casos Confirmados por Estado')





plt.xticks(rotation=0);
# Óbitos por Estado.



ax = por_estado['deaths'].plot.bar(color='C0')

ax.set_ylabel('Mortes')

#ax.set_ylim(0, 0.3)

ax.set_title('Óbitos por Estado')

plt.xticks(rotation=0);
# Dados Confirmados por cidade.



ax = por_cidade_confirmados['confirmed'].plot.barh(color='C0')

ax.set_ylabel('Cidade')

ax.invert_yaxis()

ax.set_title('Casos Confirmados por Cidade')

ax.set_xlabel("Nº Casos");



# Esse código precisa de refatoração.

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.1, i.get_y()+.31, \

            str(round((i.get_width()), 2)), color='dimgrey')



plt.xticks(rotation=0);

# Óbitos por cidade.



ax = por_cidade_obitos['deaths'].plot.barh(color='C0')

ax.set_ylabel('Cidade')

ax.invert_yaxis()

ax.set_title('Óbitos por Cidade')

ax.set_xlabel("Nº Óbitos");



# Esse código precisa de refatoração.

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.1, i.get_y()+.31, \

            str(round((i.get_width()), 2)), color='dimgrey')



plt.xticks(rotation=0);
# Por data



ax = por_data['confirmed'].plot.line(color='C0')

ax.set_ylabel('Casos')

ax.set_title('Linha do tempo - Casos Confirmados')



plt.xticks(rotation=0);
import plotly.graph_objects as go



# Plotly Express

import plotly.express as px
fig = px.line(por_data, x="date", y="confirmed",

              labels={'y':'Nº Confirmados'},

              title='Casos Confirmados')



fig.show()
fig = px.line(por_data, x="date", y="deaths",

              labels={'y':'Nº Confirmados'},

              title='Óbitos')

fig.show()