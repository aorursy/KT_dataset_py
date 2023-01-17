import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from mpl_toolkits.basemap import Basemap

%matplotlib inline

#add

import folium 

from folium import plugins

import json





import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset_cases_gps = pd.read_csv('/kaggle/input/covid19-dados-brasil/cases-gps.csv', encoding='utf-8')

casos_cidades = dataset_cases_gps[dataset_cases_gps['type'] == '1']

casos_cidades[casos_cidades.name.str.contains('/PI')]
# Set up plot

df_casos_mapa = casos_cidades

plt.figure(1, figsize=(24,12))



# Mercator of World

mapa_01 = Basemap(projection='merc',

             llcrnrlat=-35,

             urcrnrlat=7,

             llcrnrlon=-77,

             urcrnrlon=-32,

                              lat_ts=0,

             resolution='c')



mapa_01.fillcontinents(color='#f8f8f8',lake_color='#f8f8f8') # dark grey land, black lakes

mapa_01.drawmapboundary(fill_color='aqua')                # black background

mapa_01.drawcountries(linewidth=0.5, color="b")              # thin white line for country borders

mapa_01.drawstates(linewidth=0.3, color="b")

# Plot the data

mapa_01_xy = mapa_01(df_casos_mapa["lon"].tolist(), df_casos_mapa["lat"].tolist())

mapa_01.scatter(mapa_01_xy[0], mapa_01_xy[1], s=3, c="#1292db", lw=3, alpha=1, zorder=5)



plt.title("Pontos de incidência de casos confirmados")

plt.show()
#coordenadas visão Brasil

Brasil_lat = -14.235

Brasil_lon = -51.9253

mapa_Brasil = folium.Map(location=[Brasil_lat, Brasil_lon], zoom_start=4.5,tiles='cartodbpositron')

#mapa_Brasil = folium.Map(location=[br_lat, br_lon], zoom_start=4)



#Carrega a malha geográfica do Brasil

geo_json_BR = json.load(open('/kaggle/input/covid19dados-webalysson/malha_geo_br.json'))

folium.GeoJson(geo_json_BR).add_to(mapa_Brasil)





#Marcadores

for index, registro in (casos_cidades.iterrows()):

    folium.Marker(

                 location=[registro['lat'], registro['lon']], 

                 popup = (registro['name']+'<br/> Confirmados: '+str(registro['total'])), 

                 tooltip = registro['name'],

                 icon=folium.Icon(icon='info-sign')

                 ).add_to(mapa_Brasil)



mapa_Brasil
#coordenadas visão Brasil

Brasil_lat = -14.235

Brasil_lon = -39.9253

mapa_NE = folium.Map(location=[Brasil_lat, Brasil_lon], zoom_start=5,tiles='cartodbpositron')

#mapa_Brasil = folium.Map(location=[br_lat, br_lon], zoom_start=4)



#Carrega a malha geográfica do Brasil

geo_json_NE = json.load(open('/kaggle/input/covid19dados-webalysson/malha_geo_NE.json'))

folium.GeoJson(geo_json_BR).add_to(mapa_NE)



#Filtra cidades do NE

UFs_NE = ['PI','MA','CE','PE','BA','SE','AL','RN','PB']

UFs_NE = map(( lambda x: '/' + x), UFs_NE) #add '/' antes da sigla

casos_cidades_NE = casos_cidades[casos_cidades.name.str.contains('|'.join(UFs_NE))] #procura por cidades no NE



#Marcadores

for index, registro in (casos_cidades_NE.iterrows()):

    folium.Marker(

                 location=[registro['lat'], registro['lon']], 

                 popup = (registro['name']+'<br/> Confirmados: '+str(registro['total'])), 

                 tooltip = registro['name']

                 ).add_to(mapa_NE)



mapa_NE
#coordenadas visão PI

Brasil_lat = -6.00 

Brasil_lon = -43.68

mapa_PI = folium.Map(location=[Brasil_lat, Brasil_lon], zoom_start=7)

#mapa_Brasil = folium.Map(location=[br_lat, br_lon], zoom_start=4)



#Carrega a malha geográfica do PI

geo_json_PI = json.load(open('/kaggle/input/covid19dados-webalysson/malha_geo_PI_Municipios.json'))

style_function = lambda feature: {

                                'fillColor': 'green',

                                'color': 'darkred',

                                'weight': 0.1,

                               }

folium.GeoJson(geo_json_PI, 

               #style_function = style_function,

               name='Municípios'

              ).add_to(mapa_PI)





#Filtra cidades do PI

casos_cidades_PI = casos_cidades[casos_cidades.name.str.contains('PI')]

#casos_cidades_PI = casos_cidades.query("name.str.contains('PI')", engine='python')



#Marcadores

for index, registro in (casos_cidades_PI.iterrows()):

    folium.Marker(

                 location=[registro['lat'], registro['lon']], 

                 popup = (registro['name']+'<br/> Confirmados: '+str(registro['total'])), 

                 tooltip = registro['name']

                 ).add_to(mapa_PI)

mapa_PI
#Carrega o Dataset das Cidades com casos registrados e IBGEid das mesmas

casos_cidades_id = pd.read_csv('/kaggle/input/covid19-dados-brasil/cases-brazil-cities.csv')

#Filtra as Cidades do Estado do PI

casos_cidades_PI_id = casos_cidades_id[casos_cidades_id.city.str.contains('/PI')] 

#Renomeia a coluna ibgeID para codarea(atributo para unir com o Json)

casos_cidades_PI_area = casos_cidades_PI_id.rename(columns={'ibgeID': 'codarea'})

#exclui as colunas descenessárias

casos_cidades_PI_area.drop(['country', 'state', 'city'], axis=1, inplace=True)

#exibe o dataset

casos_cidades_PI_area
#Carrega o Dataset com todas as Cidades do PI e seus respectivos atributos 'codarea' do IBGE

cidades_PI_IBGE_cod = pd.read_csv('/kaggle/input/covid19dados-webalysson/cidades_PI_IBGE_Cod.csv')

#Renomeia a coluna ibgeID para codarea(atributo para unir com o Json)

cidades_PI_IBGE_cod.rename(columns={'Codigo': 'codarea'}, inplace=True)

cidades_PI_IBGE_cod.head()
#Junção dos Datasets

casos_cidades_PI_com_area = casos_cidades_PI_area.merge(cidades_PI_IBGE_cod, on='codarea', how='right')

#Substitue valores nulos por 0

casos_cidades_PI_com_area.fillna(0, inplace=True)

#converte a coluna codarea para string

casos_cidades_PI_com_area.codarea = casos_cidades_PI_com_area.codarea.astype(str)

#converte a coluna totalCases para int

casos_cidades_PI_com_area.totalCases = casos_cidades_PI_com_area.totalCases.astype('int64')

#exibe o Dataset com todas as cidades do PI e seus respectivos códigos do IBGE(codarea)

casos_cidades_PI_com_area
#coordenadas visão PI

Brasil_lat = -6.00 

Brasil_lon = -43.68

mapa_PI_Municipios_Escala = folium.Map(location=[Brasil_lat, Brasil_lon], zoom_start=7)



#mapa coroplético do Estado do PI

folium.Choropleth(

    geo_data=geo_json_PI,

    name='Piauí',

    data=casos_cidades_PI_com_area,

    columns=['codarea', 'totalCases'],

    key_on='feature.properties.codarea',

    fill_color='OrRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name='Casos no Piauí'

).add_to(mapa_PI_Municipios_Escala)

    

folium.LayerControl(collapsed=False).add_to(mapa_PI_Municipios_Escala)



#mapa_PI_Municipios_Escala.save('Mapa_PI.html')

mapa_PI_Municipios_Escala
#Para destacar municipios com ocorrencia sem considerar a escala de cores

casos_cidades_PI_com_area.loc[casos_cidades_PI_com_area.totalCases>0, 'totalCases'] = casos_cidades_PI_com_area.totalCases.max()

casos_cidades_PI_com_area
#coordenadas visão PI

Brasil_lat = -6.00 

Brasil_lon = -43.68

mapa_PI_Municipios = folium.Map(location=[Brasil_lat, Brasil_lon], zoom_start=7)



#mapa coroplético

folium.Choropleth(

    geo_data=geo_json_PI,

    name='Piauí',

    data=casos_cidades_PI_com_area,

    columns=['codarea', 'totalCases'],

    key_on='feature.properties.codarea',

    fill_color='OrRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name='Casos no Piauí',

).add_to(mapa_PI_Municipios)

    

#folium.LayerControl(collapsed=False).add_to(mapa_PI_Municipios)



#mapa_PI_Municipios.save('Mapa_PI.html')

mapa_PI_Municipios