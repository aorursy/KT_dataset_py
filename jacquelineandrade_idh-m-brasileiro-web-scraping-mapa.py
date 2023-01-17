import requests

from bs4 import BeautifulSoup

import pandas as pd

import geopandas

import matplotlib

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Requisição dos dados na página da Wikipedia sobre o IDH do Brasil



req = requests.get('https://pt.wikipedia.org/wiki/Lista_de_unidades_federativas_do_Brasil_por_IDH').text
# Criação do soup



soup = BeautifulSoup(req, 'html.parser')



#print(soup.prettify())
# Extração da primeira tabela da página do soup



table = soup.find_all('table', class_='wikitable')[0] 



#print(table.prettify())
# Conversão da tabela em uma lista



df = pd.read_html(str(table))

print(df)
# Obtenção do primeiro item da lista (que é o dataframe que será trabalhado)



df_1 = df[0].copy()

df_1
# Exclusão das colunas 0, 1 e 5



df_1 = df_1.drop(df_1.columns[[0,1,5]], axis=1)

df_1
# Mudança no nome das colunas



df_1.columns = ['UF', 'IDH-M 2017', 'IDH-M 2016']

df_1
# Deixando as palavras da coluna 'UF' em maiúscula, para ficar igual o arquivo geodataframe que

# será trabalhado em seguida



df_1['UF'] = df_1['UF'].str.upper()

df_1
# Importando o geodataframe com o mapa do Brasil dividido em estados



#print(os.listdir('../input')) #ver quais arquivos estão importados



brazil = geopandas.read_file("../input/estados-br-geopandas/BRUFE250GC_SIR.shp")

brazil
# Renomeando as colunas do geodataframe 'Brazil', para não haver problema no merge a seguir



brazil.columns = ['UF', 'Região', 'CD_GEOCUF', 'geometry']

brazil
# União dos dados extraídos da Wikipedia e o geodataframe 'Brazil'



df_geo = pd.merge(df_1, brazil[['UF', 'Região', 'geometry']], on='UF')

df_geo
# Transformando o dataframe anterior em um geodataframe (que será plotado futuramente

# com o mapa do Brasil (o geodataframe 'brazil'))



gdf = geopandas.GeoDataFrame(df_geo)



gdf
# CÓDIGO DO MAPA



# Detalhes do mapa



fig, ax = plt.subplots(1, 1, figsize=(9, 9)) # tamanho da figura

ax.set_facecolor('#ffffff') # cor do fundo do mapa



# Código da legenda



from mpl_toolkits.axes_grid1 import make_axes_locatable



divider = make_axes_locatable(ax)

cax = divider.append_axes("bottom", size="10%", pad=0.1) # localização da barrinha e legenda



# Código das cores personalizadas da barra da legenda

# (as cores podem ser cores padrão do matplotlib ou um hex)

# (cria uma escala de cores entre as cores escolhidas)



cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ff8585", "#6ed16b", "#1da118", "#1e207d"])



# Define os limites da barra da legenda e normaliza os valores

# (o cmap.N faz com que todas as cores da escala apareçam)



norm = matplotlib.colors.BoundaryNorm([600,700,800,900,1000], cmap.N)



# Criar um plot entre geodataframes (o 'brazil' e o criado com os dataframes ('gdf'))



# (plot com o geodataframe base ('brazil'))

brazil.plot(ax=ax, facecolor='#d4d4d4', edgecolor='#ffffff', linewidth=0.4)



# (plot com o geodataframes com informações ('gdf'))

gdf.plot(column='IDH-M 2017', ax=ax, legend=True, cax=cax, cmap=cmap, edgecolor='#ffffff', linewidth=0.4,

        legend_kwds={'label': "IDM-H em 2017 por Unidade da Federação", 'orientation': "horizontal"},

         norm=norm)