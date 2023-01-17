# from mpl_toolkits.mplot3d import Axes3D

# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap
dirname = '../input/covid19-colombia-httpswwwdatosgovco/rows.csv'

data_covid_prev = pd.read_csv(dirname, index_col='ID de caso')

data_covid=data_covid_prev

# data_covid[data_covid['Fecha de muerte']=='-   -']

data_covid['Fecha de muerte']=data_covid['Fecha de muerte'].replace('-   -','')

data_covid['Fecha de muerte'].unique()
# dirname = '../input/covid19-colombia-httpswwwdatosgovco/rows.csv'

# data_covid_prev = pd.read_csv(dirname, index_col='ID de caso')

# data_covid=data_covid_prev



# # Ajustando los DataTime



data_covid['fecha reporte web']=pd.to_datetime(data_covid_prev['fecha reporte web'])

data_covid['Fecha recuperado']=pd.to_datetime(data_covid_prev['Fecha recuperado'])

data_covid['Fecha de muerte']=pd.to_datetime(data_covid_prev['Fecha de muerte'])

# data_covid['FIS']=pd.to_datetime(data_covid['FIS'])

data_covid['fecha reporte web']=pd.to_datetime(data_covid_prev['fecha reporte web'])



# Faltan datos de diagnostico, hay nan

data_covid[data_covid['fecha reporte web'].isnull()]

data_covid['fecha reporte web']=data_covid['fecha reporte web'].fillna(method='ffill')

data_covid0=data_covid.copy()

data_covid_alt=data_covid.copy()


# data_covid['Fecha diagnostico']=data_covid['Fecha diagnostico'].replace('Sin dato', '2020-05-29')

data_covid.index=pd.to_datetime(data_covid['fecha reporte web'],format='%Y-%m-%d')



data_covid=data_covid.sort_index()

data_covid[data_covid.index.notna()].tail()
pG=data_covid.replace([np.nan,'nan'],np.nan).loc[:,['Departamento o Distrito ','fecha reporte web','Fecha recuperado','Fecha de muerte']].groupby(['Departamento o Distrito '])

aux=lambda p: sum((p['Fecha recuperado']).map(lambda pe: len(pe)>1))

# pG.apply(lambda p: sum((p['Fecha recuperado']).map(lambda pe: len(pe)>1)))

pG2=pG.apply(lambda p: p.count())

# pG.agg(['min','max'])

# ##############



pG3=pd.DataFrame()

pG3['Enfermos']=pG2['Departamento o Distrito ']-(pG2['Fecha recuperado']+pG2['Fecha de muerte'])

pG3['Recuperados']=pG2['Fecha recuperado']

pG3['Muertes']=pG2['Fecha de muerte']



index_sorted=pG3.sum(axis=1).sort_values(ascending=False)



pG3.sort_values(by='Enfermos',ascending=True).plot(kind='barh',figsize = (12,8),stacked=True)
pG4=pG3.div(pG3.sum(axis=1), axis=0)

pG4.sort_values(by='Enfermos',ascending=True).plot(kind='barh',figsize = (12,8),stacked=True)

plt.title('Tasas: actualmente Enfermos vs Recuperados vs Muertos')

pG5=pG4

pG5['Total']=pG3.sum(axis=1)
plt.figure(figsize=(16,50))

valores=pG5.sort_values(by='Enfermos',ascending=True)

anchura=valores.Total

plt.barh((anchura.rolling(2).sum().fillna(2)*0.5+2000).cumsum(),height=anchura,width=valores.Enfermos,tick_label=valores.index)

plt.barh((anchura.rolling(2).sum().fillna(2)*0.5+2000).cumsum(),height=anchura,left=valores.Enfermos,width=valores.Recuperados)#,tick_label=valores.index

plt.barh((anchura.rolling(2).sum().fillna(2)*0.5+2000).cumsum(),height=anchura,left=valores.Enfermos+valores.Recuperados,width=valores.Muertes)#,tick_label=valores.index



for ith in range(len(valores)):

    fS=int(anchura[ith])//1500

#     print(fS)

    plt.text(0.5,(anchura.rolling(2).sum().fillna(2)*0.5+2000).cumsum()[ith],'{}'.format(str(valores.index[ith])),fontsize=fS,horizontalalignment='center') 

plt.show()
infectados_ubicacion=pd.DataFrame(data_covid['Ciudad de ubicación'].value_counts())

infectados_ubicacion2=infectados_ubicacion.sort_values(by='Ciudad de ubicación',ascending=True)

infectados_ubicacion.iloc[[9,8,7,6,5,4,3,2,1,0]].plot(kind='barh',figsize = (12,8))



plt.title('Figura 1. Afectados por ciudad')
# https://requests.readthedocs.io/es/latest/

import requests



url="https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json"

# url="https://raw.githubusercontent.com/finiterank/mapa-colombia-js/master/colombia-municipios.json"

data = requests.get(url)



data_geo=gpd.GeoDataFrame().from_features(data.json())

# data_geo=data.json()['objects']['mpios']



# data_geo.PERIMETER.index=data_geo.PERIMETER.index.map(lambda a : str(a))

pG9=pG2

pG9.loc['Atlántico']=pG9.loc['Atlántico']+pG9.loc['Barranquilla D.E.']

pG9.loc['Cundinamarca']=pG9.loc['Cundinamarca']+pG9.loc['Bogotá D.C.']

pG9.loc['Valle del Cauca']=pG9.loc['Valle del Cauca']+pG9.loc['Buenaventura D.E.']

pG9.loc['Magdalena']=pG9.loc['Magdalena']+pG9.loc['Santa Marta D.T. y C.']

pG9.loc['Bolívar']=pG9.loc['Bolívar']+pG9.loc['Cartagena D.T. y C.']
data_geo['new']=[pG9.loc[data_covid.groupby(['Codigo departamento']).first().loc[int(data_geo.DPTO[i])]['Departamento o Distrito ']][0] for i in range(len(data_geo.DPTO))]

data_geo.index=data_geo.index.map(lambda a : str(a))

# Faltan los "departamentos" que pucieron como ciudades D.C
# Create a base map

m = folium.Map(location=[4.570868 , -74.297333], tiles='cartodbpositron', zoom_start=5.5)



# Add a choropleth map to the base map

Choropleth(geo_data=data_geo.geometry.__geo_interface__, 

           data=data_geo.new, 

           key_on="feature.id", 

           fill_color='YlOrRd', 

           legend_name='Casos de covid-19'

          ).add_to(m)



# Display the map

m
# data_covid_alt=data_covid0[data_covid0['Departamento o Distrito ']=='Antioquia']
new_0=data_covid_alt.groupby(['fecha reporte web'])['Estado'].value_counts()

new_0=new_0.unstack()

new_0=new_0.fillna(0)

# new_0

graf2=new_0.div(new_0.sum(axis=1), axis=0)

graf2=graf2.loc[:,['Asintomático','Leve','Moderado','Grave','Fallecido']]

# graf2.plot(kind='area',stacked=True,alpha=0.1)
total_x_diag=data_covid_alt.groupby(['fecha reporte web'])['fecha reporte web'].count()

recup_x_diag=data_covid_alt.groupby(['fecha reporte web'])['Fecha recuperado'].apply(lambda p : sum(~np.isnan(p.values)))

muert_x_diag=data_covid_alt.groupby(['fecha reporte web'])['Fecha de muerte'].apply(lambda p : sum(~np.isnan(p.values)))

# plt.figure()

graf=pd.DataFrame()

graf['Muertes']=muert_x_diag

graf['Recuperados']=recup_x_diag

graf['Actualmente']=total_x_diag-(recup_x_diag+muert_x_diag)

graf[graf.index!='nan'].plot(kind='bar',stacked=True,figsize=(16,6))

# graf2.plot(kind='area',stacked=True,alpha=0.6,figsize=(16,2))

plt.plot(0, 0, label='First Line')
Data_Diagnosticados=pd.Series(data_covid['fecha reporte web'].value_counts())

# muertos

Data_Muertos=pd.Series(data_covid['Fecha de muerte'][1:].value_counts())

# Recuperados

Data_Recuperados=pd.Series(data_covid['Fecha recuperado'].value_counts())

# Data_Muertos



Data_Diagnosticados_acum=Data_Diagnosticados.sort_index(ascending=True).cumsum()

Data_Muertos_acum=Data_Muertos[1:].sort_index(ascending=True).cumsum()

Data_Recuperados_acum=Data_Recuperados[1:].sort_index(ascending=True).cumsum()

df = pd.concat([Data_Muertos_acum,Data_Recuperados_acum,Data_Diagnosticados_acum], axis=1,sort=True)

df.plot(figsize=(12,8),stacked=False)

plt.title('Figura 3. Casos acumulados de muertes, recuperados y diagnosticados')

df.to_csv('pruebaSave.csv')
cols=data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)].columns

(data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[3]].value_counts()).sort_values(ascending=True).plot.barh(figsize=(14,8)) #principales muertos por ciudad

muerte_edad=data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[5]].value_counts()/data_covid[cols[5]].value_counts() #principales muertos por ciudad

plt.title('Figura 4. Muertos totales por ciudad')

plt.show()

muerte_edad.sort_index(ascending=True).plot.bar(figsize=(14,8))



plt.title('Figura 4.5. Muertes por edad')

data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[5]].value_counts().sort_index(ascending=True).plot.bar(figsize=(14,8))

plt.show()



plt.title('Figura 5. Ratio de muertes por edad')

muerte_edad.plot.bar(figsize=(14,8))

plt.show()



plt.title('Figura 5.1. Muertes importado/relacionado')

data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[7]].value_counts().plot.bar(figsize=(14,8))

plt.show()



plt.title('Figura 5.2 Muertes según el sexo')

data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[6]].value_counts().plot.bar(figsize=(14,8))

plt.show()



((data_covid[data_covid['Fecha de muerte'].map(lambda p: len(str(p)) > 8)][cols[3]].value_counts()/data_covid[cols[3]].value_counts()).sort_values(ascending=True)).plot.barh(figsize=(14,8))

plt.title('Figura 7: Porcentaje de muertos por departamentos')
# print('Solo 4 personas se han registrado como no tienen fecha de Primeros sintomas...\n',(data_covid['FIS']).map(lambda p: len(p)>8).value_counts())

# print('Se han diagnosticado más de 2000 como asintomáticas...\n',(data_covid['Estado'].value_counts()))

(data_covid['Estado'].value_counts().sort_values(ascending=True).plot.barh(figsize=(14,8)))
for ciudad_n in range(len(data_covid['Ciudad de ubicación'].unique())):

# ciudad_n=7

    print(sorted(data_covid['Ciudad de ubicación'].unique())[ciudad_n])

    data_covid[data_covid['Ciudad de ubicación']==sorted(data_covid['Ciudad de ubicación'].unique())[ciudad_n]]['fecha reporte web'].value_counts().sort_index(ascending=True).plot(figsize=(12,3),kind='bar')

    plt.title(sorted(data_covid['Ciudad de ubicación'].unique())[ciudad_n])

    plt.show()
for departamento_n in range(len(data_covid['Departamento o Distrito '].unique())):

#     departamento_n=7

    print(sorted(data_covid['Departamento o Distrito '].unique())[departamento_n],'\r\n')

    data_covid[data_covid['Departamento o Distrito ']==sorted(data_covid['Departamento o Distrito '].unique())[departamento_n]]['fecha reporte web'].value_counts().sort_index(ascending=True).plot(figsize=(16,3),kind='bar')

#     data_covid[data_covid['Departamento o Distrito ']==data_covid['Departamento o Distrito '].unique()[departamento_n]]['Fecha diagnostico'].value_counts().sort_values(ascending=True).cumsum().plot(figsize=(12,8))

    plt.title(sorted(data_covid['Departamento o Distrito '].unique())[departamento_n])

    plt.show()
data_covid[data_covid['Ciudad de ubicación']=='Montería']['Edad'].value_counts().sort_index().plot(kind='bar',figsize=(16,8))

# plt.title('Montería')

# plt.show()
data_covid[data_covid['Ciudad de ubicación']=='Montería'][data_covid[data_covid['Ciudad de ubicación']=='Montería']['Fecha de muerte']!='nan']['Edad'].value_counts().sort_index().plot(kind='bar',figsize=(16,8))
