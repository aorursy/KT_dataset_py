import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))

import folium

# Base_1 -> Dados sobre Barragens brasileiras

main_file_path = '../input/database_versao_LatLongDecimal_fonteANM_23_01_2019.csv' 

base_1 = pd.read_csv(main_file_path)

base_1.isnull().sum().sort_values(ascending = False).head(10)



## Plotando o Grafico 1 -> qntd Barragens X Minério Principal

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='UF', data=df_barragens, order=df_barragens["UF"].value_counts().index)

base_1['MINERIO_PRINCIPAL'].value_counts()[:10].plot(kind="barh", color="green")

plt.title('Grafico 1 - Quantidade de Barragens por Minério Principal para as 10 atividades principais')

plt.show()
## Tratando os dados ##

#EXCLUIR AS COLUNAS NOME BARRAGEM MINERACAO, CPF, POSICIONAMENTO

base_1.drop(['NOME_BARRAGEM_MINERACAO','CPF_CNPJ','POSICIONAMENTO'], axis=1, inplace=True)



#Encontrando valores faltantes

base_1.isnull().sum().sort_values(ascending = False).head(10)



base_1.head(10)
## Plotando o Grafico 2 -> qntd Barragens X UF

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='UF', data=df_barragens, order=df_barragens["UF"].value_counts().index)

base_1['UF'].value_counts().plot(kind="bar", color="green")

plt.title('Gráfico 2 - Número de Barragens de Resíduos e Rejeito por Unidade da Federação')

plt.show()
# Visualizacao 1 - Mapa da localizacao das barragens com Categoria de Risco == "Alta" e das barragens com 

#cria o mapa

mapa = folium.Map(location=[-14.235, -51.9253], zoom_start=5, tiles='Stamen Terrain')



######################################################################################################################################

##Vamos plotar as barragens com Sem Categoria de Risco e Sem Dano Potencial em lARANJA no Mapa



#seleciona dados das barragens sem categoria de risco

semrisco = base_1[base_1['INSERIDA_NA_PNSB'] == 'Não']

municipios =semrisco.MUNICIPIO

minerios =semrisco.MINERIO_PRINCIPAL

latitudes =semrisco.LATITUDE

longitudes =semrisco.LONGITUDE



#plota os dados das barragens sem categoria de risco no mapa na cor laranja

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='orange', icon='info-sign')).add_to(mapa)

####################################################################################################################################



#######################################################################################################################################

##Vamos plotar as barragens com Dano Potencial Associado alto em Preto no Mapa



#seleciona dados das barragens de DANO_POTENCIAL_ASSOCIADO alto

altorisco = base_1[base_1['DANO_POTENCIAL_ASSOCIADO'] == 'Alta']

municipios =altorisco.MUNICIPIO

minerios =altorisco.MINERIO_PRINCIPAL

latitudes =altorisco.LATITUDE

longitudes =altorisco.LONGITUDE



#plota os dados das barragens com Dano Potencial Associado alta no mapa na cor preta

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='black', icon='info-sign')).add_to(mapa)

    

#####################################################################################################################################    

##Vamos plotar as barragens com Categoria de Risco alto em Vermelho no Mapa

    

#seleciona dados das barragens de  categoria de risco alto

altorisco = base_1[base_1['CATEGORIA_DE_RISCO'] == 'Alta']

municipios =altorisco.MUNICIPIO

minerios =altorisco.MINERIO_PRINCIPAL

latitudes =altorisco.LATITUDE

longitudes =altorisco.LONGITUDE



#plota os dados das barragens com Categoria de Risco alta no mapa na cor preta

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='red', icon='info-sign')).add_to(mapa)
mapa
## Plotando o Grafico 3 -> Numero de Barragens X Municipio

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='MUNICIPIO', data=df_barragens, order=df_barragens["MUNICIPIO"].value_counts().index)

base_1['MUNICIPIO'].value_counts()[0:14].plot(kind="bar", color="green")

plt.title('Grafico 3 - 10 Cidades com mais barragens no Brasil')

plt.show()
# Base_2 -> Dados sobre Barragens brasileiras -> POCONÉ

main_file_path = '../input/database_versao_LatLongDecimal_fonteANM_23_01_2019.csv' 

base_2 = pd.read_csv(main_file_path)

base_2[base_2['MUNICIPIO'] == 'POCONÉ']
## Plotando o Grafico 4 -> Risco Associado X UF

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='DANO_POTENCIAL_ASSOCIADO', data=df_barragens, order=df_barragens["DANO_POTENCIAL_ASSOCIADO"].value_counts().index)

base_1['CATEGORIA_DE_RISCO'].value_counts().plot(kind="barh", colormap = "Accent" )

plt.title('Grafico 4 - Categoria de Risco')

plt.show()
## Plotando o Grafico 5 -> Risco Associado X UF

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='DANO_POTENCIAL_ASSOCIADO', data=df_barragens, order=df_barragens["DANO_POTENCIAL_ASSOCIADO"].value_counts().index)

base_1['DANO_POTENCIAL_ASSOCIADO'].value_counts().plot(kind="barh", colormap = "Accent" )

plt.title('Grafico 5 - Dano Potencial Associado')

plt.show()
## Plotando o Grafico X -> Aderencia PNSB X QNTD DE Barragens

plt.figure(1 , figsize = (12 , 5))

# sns.countplot(x='UF', data=df_barragens, order=df_barragens["UF"].value_counts().index)

base_1['INSERIDA_NA_PNSB'].value_counts("Sim").plot(kind="pie")

plt.title('Grafico 6 - Porcentagem de Barragens de Resíduos e Rejeito Aderentes ao PNSB')

plt.show()