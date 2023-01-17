# Importando as bibliotecas que serão utilizadas:

import os
import shapefile
from json import dumps
import numpy as np
import pandas as pd
import geopandas as gpd
import descartes as dc
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from branca.colormap import linear
from geopandas import GeoSeries
import shapely
import json
import plotly.express as px
# Inicializando o Visualization Set:

sns.set(style='whitegrid', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10,6))
# Elencando todos os arquivos disponíveis no diretório:

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Lendo os arquivos necessários e abrindo o mapa vetor:

gdf1 = gpd.read_file('/kaggle/input/classificao-municpios-enem-2018/BR_Municipios_2019.shp')

sf = gpd.GeoDataFrame(gdf1)
# Verificando o tamanho do Geodataframe gerado:

sf.shape
# Visualizando o shapefile:

sf.head(5)
# Gerando um mapa limpo:

sf.plot(figsize=(20,20), linewidth=0.3)
# Lendo o arquivo em formato Geopandas:

df = gpd.read_file('/kaggle/input/classificao-municpios-enem-2018/ENEM_2018_Classificacao_Municipios.csv')
# Verificando uma amostra do dataset lido:

df.sample(5)
# Verificando o tamanho do Geodataframe gerado:

df.shape
# Retirar as colunas com informações que não serão relevantes para a análise:

df = df.drop(['SG_UF_RESIDENCIA', 'NO_MUNICIPIO_RESIDENCIA', 'NOME_MUNICIPIO', 'NU_NOTA_CN_Mean', 'NU_NOTA_CH_Mean', 'NU_NOTA_LC_Mean', 
              'NU_NOTA_MT_Mean', 'NU_NOTA_REDACAO_Mean', 'NU_NOTA_CN_StdDev', 'NU_NOTA_CH_StdDev', 'NU_NOTA_LC_StdDev', 'NU_NOTA_MT_StdDev', 
              'NU_NOTA_REDACAO_StdDev', 'NU_NOTA_CN_Var','NU_NOTA_CH_Var', 'NU_NOTA_LC_Var', 'NU_NOTA_MT_Var', 'NU_NOTA_REDACAO_Var', 'NU_NOTA_CN_N', 
              'NU_NOTA_CH_N', 'NU_NOTA_LC_N', 'NU_NOTA_MT_N', 'NU_NOTA_REDACAO_N', 'NU_NOTA_CN_Median', 'NU_NOTA_CH_Median', 'NU_NOTA_LC_Median', 
              'NU_NOTA_MT_Median', 'NU_NOTA_REDACAO_Median', 'NU_NOTA_CN_Q3', 'NU_NOTA_CH_Q3', 'NU_NOTA_LC_Q3', 'NU_NOTA_MT_Q3', 'NU_NOTA_REDACAO_Q3', 
              'NU_NOTA_CN_P90', 'NU_NOTA_CH_P90', 'NU_NOTA_LC_P90', 'NU_NOTA_MT_P90', 'NU_NOTA_REDACAO_P90', 'Distance', 'geometry', 'LATITUDE', 'LONGITUDE'], axis=1)
df.head()
# Verificando os tipos de dados que compõem o dataframe:

df.dtypes
# Convertendo os valores da coluna 'Segment Id' de object para int:

df['Segment Id'] = df['Segment Id'].astype(object).astype(int)
# Confirmando a conversão:
df.dtypes
# Concatenando os datasets:

merged = sf.set_index('CD_MUN').join(df.set_index('CO_MUNICIPIO_RESIDENCIA'))
merged.head()
# Verificando os tipos de dados que compõem o novo dataframe:

merged.dtypes
# plotando o mapa do Brasil, com todos os seus municípios e com a identificação, via mapa de calor, de acordo com a id de cada município segundo o arquivo .csv:

# plota estimativas de população / dados com uma legenda mais acurada
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, figsize=(20, 20), linewidth=0.3)

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="3%", pad=0.1)

# Adicionando um título
ax.set_title('Classificação Enem - Brasil', color='#555555', fontdict={'fontsize': '30', 'fontweight' : '3'})

# Criando uma nota de rodapé
ax.annotate('Fontes: Microdados do Enem (INEP) e Malha Municipal (IBGE)', xy=(0.1, .09), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=15, color='#555555')

merged.plot(column='Segment Id', ax=ax, cmap='YlOrRd', linewidth=0.2, legend=True, cax=cax)

plt.rcParams['axes.facecolor'] = 'lightgray'
# Salvando o mapa
fig.savefig('Brasil_enem.png', dpi=600)   
