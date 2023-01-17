import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import folium
barr = pd.read_csv('../input/database_versao_LatLongDecimal_fonteANM_23_01_2019.csv', sep=',')

barr.head()
plt.figure(figsize=(16,8))

plt.title('Quantidade de barragens por estado(UF)', )

plt.ylabel('Sigla dos estados.')

barr['UF'].value_counts().plot(kind='barh', color='blue' )
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
uf = barr['UF'].value_counts()[:10]

data = [go.Bar(x=uf.values, y=uf.index, orientation='h', name='Quantidade:'),

        go.Bar(x=[barr.shape[0]], y=['Total'], orientation='h', name='Total de barragens.')

        ]

layout = go.Layout(title="Dez estados com mais barragens.",            

                    yaxis=dict(

                            showgrid=False,

                            showline=False,

                            showticklabels=True,

                            zeroline=False,

                            title="Estados(UF)",

                            ),

                  )



fig = go.Figure(data=data, layout=layout)

iplot(fig)
minerios = barr['MINERIO_PRINCIPAL'].value_counts()[:10]

data = [go.Bar(x=minerios.values, y=minerios.index, orientation='h', name='Quant. Barragens:'),

        go.Bar(x=[barr.shape[0]], y=['Total'], orientation='h', name='Total de barragens.')

        ]



layout = go.Layout(title="Dez minerios mais explorados nas barragens.",

                    yaxis=dict(

                            showgrid=True,

                            showline=True,

                            showticklabels=True,

                            zeroline=True,

                            ),

                      )

fig = go.Figure(data=data, layout=layout)

iplot(fig)
municipios = barr['MUNICIPIO']+"/"+barr['UF']

m = municipios.value_counts()[:10]

data = [go.Bar(x=m.values, y=m.index, orientation='h', name='Quant. Barragens:'),

            ]

go.layout.title.font ={"family":"Times New Roman", 'size':16, 'color':'red'}



layout = go.Layout(title="Dez cidades com mais barragens no Brasil.",

                   

                   yaxis={'showgrid':True,

                            'showline':True,

                            'showticklabels':True,

                            'zeroline':True,

                              'title':'Cidades'}

                            ,

                   xaxis={'title':'Quantidades:'}

                      )

fig = go.Figure(data=data, layout=layout, )

iplot(fig)
plt.figure(figsize=(14,10))

plt.title("Categoria de risco das barragens.")

plt.ylabel("Quantidade")

plt.xlabel("Categorias")

barr['CATEGORIA_DE_RISCO'].value_counts().plot(kind='bar')
plt.figure(figsize=(14,10))

plt.title("Dano potencial associado as barragens.")

plt.ylabel("Quantidade")

plt.xlabel("Categorias")

barr['DANO_POTENCIAL_ASSOCIADO'].value_counts().plot(kind='bar')
plt.figure(figsize=(14,10))

plt.title("Barragens de risco por estado.")

barr.groupby(['CATEGORIA_DE_RISCO'])['UF'].value_counts().plot(kind='bar')
plt.figure(figsize=(14,10))

plt.title("Dano potencial das barragens por estado.")

barr.groupby(['DANO_POTENCIAL_ASSOCIADO'])['UF'].value_counts().plot(kind='bar')
#cria o mapa

mapa = folium.Map(location=[-14.235, -51.9253], zoom_start=5, tiles='Stamen Terrain')



#seleciona dados das barragens de risco baixo

barragens_baixa = barr[barr['CATEGORIA_DE_RISCO'] == 'Baixa']

municipios =barragens_baixa.MUNICIPIO

minerios =barragens_baixa.MINERIO_PRINCIPAL

latitudes =barragens_baixa.LATITUDE

longitudes =barragens_baixa.LONGITUDE



#plota os dados das barragens de risco baixo no mapa na cor verde

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='green', icon='info-sign')).add_to(mapa)



#seleciona dados das barragens de risco médio

barragens_media = barr[barr['CATEGORIA_DE_RISCO'] == 'Média']

municipios =barragens_media.MUNICIPIO

minerios =barragens_media.MINERIO_PRINCIPAL

latitudes =barragens_media.LATITUDE

longitudes =barragens_media.LONGITUDE



#plota os dados das barragens de risco médio no mapa na cor azul

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='blue', icon='info-sign')).add_to(mapa)



#seleciona dados das barragens de risco alto

barragens_alta = barr[barr['CATEGORIA_DE_RISCO'] == 'Alta']

municipios =barragens_alta.MUNICIPIO

minerios =barragens_alta.MINERIO_PRINCIPAL

latitudes =barragens_alta.LATITUDE

longitudes =barragens_alta.LONGITUDE



#plota os dados das barragens de risco alto no mapa na cor vermelha

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='red', icon='info-sign')).add_to(mapa)





#seleciona dados das barragens sem categoria de risco

barragens_nan = barr[barr['CATEGORIA_DE_RISCO'] == 'NaN']

municipios =barragens_alta.MUNICIPIO

minerios =barragens_alta.MINERIO_PRINCIPAL

latitudes =barragens_alta.LATITUDE

longitudes =barragens_alta.LONGITUDE



#plota os dados das barragens sem categoria de risco no mapa na cor preta

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='black', icon='info-sign')).add_to(mapa)

mapa
#cria o mapa

mapa = folium.Map(location=[-14.235, -51.9253], zoom_start=5)



#seleciona dados das barragens de DANO_POTENCIAL_ASSOCIADO baixo

barragens_baixa = barr[barr['DANO_POTENCIAL_ASSOCIADO'] == 'Baixa']

municipios =barragens_baixa.MUNICIPIO

minerios =barragens_baixa.MINERIO_PRINCIPAL

latitudes =barragens_baixa.LATITUDE

longitudes =barragens_baixa.LONGITUDE



#plota os dados das barragens de DANO_POTENCIAL_ASSOCIADO baixo no mapa na cor verde

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='green', icon='info-sign')).add_to(mapa)



#seleciona dados das barragens de DANO_POTENCIAL_ASSOCIADO médio

barragens_media = barr[barr['DANO_POTENCIAL_ASSOCIADO'] == 'Média']

municipios =barragens_media.MUNICIPIO

minerios =barragens_media.MINERIO_PRINCIPAL

latitudes =barragens_media.LATITUDE

longitudes =barragens_media.LONGITUDE



#plota os dados das barragens de risco médio no mapa na cor azul

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='blue', icon='info-sign')).add_to(mapa)



#seleciona dados das barragens de DANO_POTENCIAL_ASSOCIADO alto

barragens_alta = barr[barr['DANO_POTENCIAL_ASSOCIADO'] == 'Alta']

municipios =barragens_alta.MUNICIPIO

minerios =barragens_alta.MINERIO_PRINCIPAL

latitudes =barragens_alta.LATITUDE

longitudes =barragens_alta.LONGITUDE



#plota os dados das barragens de DANO_POTENCIAL_ASSOCIADO alto no mapa na cor vermelha

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='red', icon='info-sign')).add_to(mapa)





#seleciona dados das barragens sem categoria de DANO_POTENCIAL_ASSOCIADO

barragens_nan = barr[barr['DANO_POTENCIAL_ASSOCIADO'] == 'NaN']

municipios =barragens_alta.MUNICIPIO

minerios =barragens_alta.MINERIO_PRINCIPAL

latitudes =barragens_alta.LATITUDE

longitudes =barragens_alta.LONGITUDE

#plota os dados das barragens sem categoria de DANO_POTENCIAL_ASSOCIADO no mapa  na cor preta

for municipio, minerio, latitude, longitude in zip(municipios, minerios, latitudes, longitudes):

    folium.Marker(location=[latitude, longitude], popup=minerio, tooltip=municipio,

              icon=folium.Icon(color='black', icon='info-sign')).add_to(mapa)



mapa