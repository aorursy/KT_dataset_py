#Alumnos:

#Bobadilla Catalan, German - 90123

#Briglia, Antonella - 90903

#Calvani, Sergio Alejandro - 98588

#Fernandez Pandolfo, Franco - 100467
#Importamos las librerias

import pandas as pd

import geopandas as gpd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import datetime as datetime

from shapely.geometry import Point

from matplotlib.colors import ListedColormap



%matplotlib inline

plt.style.use('default')

sns.set(style="whitegrid")
#Cargamos el csv

propiedades=pd.read_csv('../input/train.csv')
#Cargamos el .shp de México

#Link de donde se descargo: https://map.igismap.com/maps/layer_view/work20180914135320:MexicoPoly02/786

url="../input/MexicoPoly02.shp"

mex=gpd.read_file(url)
#Cargamos el .shp de los municipios de México

#Link de donde se descargo: http://www.conabio.gob.mx/informacion/metadata/gis/muni_2012gw.xml?_httpcache=yes&_xsl=/db/metadata/xsl/fgdc_html.xsl&_indent=no

url_municipio="../input/Muni_2012gw.shp"

mex_municipio=gpd.read_file(url_municipio)
#Establecemos las longitudes extremas de México

lat_maxima=32.718333

lat_minima=14.540833

lng_maxima=-86.710000

lng_minima=-118.366666
#Creamos DataFrame con las capitales de las Provincias de México

coordenadas_capiteles=[['Aguascalientes',21.880833, -102.296111], ['Mexicali',32.663333,-115.467778], ['La Paz ',24.142222,-110.310833], ['San Francisco de Campeche',19.848611,-90.525278], ['Tuxtla Gutiérrez',16.753056, -93.115556], ['Chihuahua',28.635278,-106.088889], ['Saltillo',25.433333,-101], ['Colima',19.243611,-103.730833], ['Victoria de Durango',24.022778,-104.654444], ['Guanajuato',21.017778,-101.256667], ['Chilpancingo de los Bravo',17.551389,-99.500833], ['Pachuca de Soto',20.1225, -98.736111], ['Guadalajara',20.676667,-103.3475], ['Toluca de Lerdo',19.292222,-99.653889], ['Morelia',19.768333,-101.189444], ['Cuernavaca',18.918611,-99.234167], ['Tepic',21.5,-104.9], ['Monterrey',25.671389,-100.308611], ['Oaxaca de Juárez',17.083333, -96.75], ['Puebla de Zaragoza',19.051389,-98.217778], ['Santiago de Querétaro',20.588056,-100.388056], ['Chetumal',18.503611,-88.305278], ['San Luis Potosí',22.149722,-100.975], ['Culiacán',24.8,-107.383333], ['Hermosillo',29.095556,-110.950833], ['Villahermosa',17.986944,-92.919444], ['Ciudad Victoria',23.736111,-99.146111], ['Tlaxcala de Xicohténcatl',19.31695,-98.238231], ['Xalapa-Enríquez',19.54,-96.9275], ['Mérida',20.97, -89.62], ['Zacatecas',22.771667,-102.575278], ['Ciudad de Mexico',19.419444,-99.145556]]

capitales=pd.DataFrame(coordenadas_capiteles,columns=['Nombre','Lat','Long'])

#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(capitales['Long'], capitales['Lat'])]

capitales_provincia = gpd.GeoDataFrame(capitales, geometry=geometry)

#Nos quedamos solamente con la Ciudad de México para utilizarlo después

ciudad_mexico=capitales_provincia[(capitales_provincia['Nombre']=='Ciudad de Mexico')]
#Vista previa del set de Propiedades

propiedades.head()
#Vista previa del .shp de México

mex.head()
#Vista previa del .shp de los municipios de México

mex_municipio.head()
#Vista previa del DataFrame de las capitales de las provincias

capitales_provincia.head()
#Renombramos columnas para poder mergear luego

mex_municipio.rename(columns={'CVE_ENT':'provincia','NOM_MUN':'ciudad'},inplace=True)
#Renombramos columnas y el nombre de algunas provincias para que tengan el mismo nombre que en el set de Datos

mex['locname'].replace({'Baja California':'Baja California Norte','Coahuila de Zaragoza':'Coahuila','Estado de México':'Edo. de México','Ciudad de México':'Distrito Federal','Michoacán de Ocampo':'Michoacán','Veracruz de Ignacio de la Llave':'Veracruz','San Luis Potosí':'San luis Potosí'},inplace=True)

mex.rename(columns={'locname':'provincia'},inplace=True)
#Observamos el tipo de dato que es cada columna y,así mismo, la cantidad de memoria que ocupa

propiedades.info()
#Cambiamos el tipo de dato de algunas columnas para bajar el uso de memoria

propiedades['antiguedad']=propiedades['antiguedad'].astype('int32',errors='ignore')

propiedades['habitaciones']=propiedades['habitaciones'].astype('int32',errors='ignore')

propiedades['banos']=propiedades['banos'].astype('int32',errors='ignore')

propiedades['gimnasio']=propiedades['gimnasio'].astype('int32',errors='ignore')

propiedades['usosmultiples']=propiedades['usosmultiples'].astype('int32',errors='ignore')

propiedades['piscina']=propiedades['piscina'].astype('int32',errors='ignore')

propiedades['escuelascercanas']=propiedades['escuelascercanas'].astype('int32',errors='ignore')

propiedades['centroscomercialescercanos']=propiedades['centroscomercialescercanos'].astype('int32',errors='ignore')
#Verificamos que efectivamente se hicieron los cambios

propiedades.info()
#Verificamos si hay algún elemento nulo en las columnas, lo cual efectivamente es cierto

propiedades.isnull().any()
#Vemos cuantos elementos nulos hay por columna

propiedades.isnull().sum()
#Verificamos si hay datos duplicados (no hay)

duplicated = propiedades.duplicated()

duplicated.value_counts()
#No se observan valores minimos y maximos irreales

propiedades.describe()
#Hacemos la conversión de fechas

propiedades['Fecha']=pd.to_datetime(propiedades['fecha'])

propiedades.drop(columns={'fecha'},inplace=True)
#Generamos las columnas de Año, Mes y Dia

propiedades['Anio']=propiedades['Fecha'].dt.year

propiedades['Mes']=propiedades['Fecha'].dt.month

propiedades['Dia']=propiedades['Fecha'].dt.day

propiedades.head()
#Creamos una función que según el mes y el dia, nos devuelve en que estación se encuentra

def fechaEstacion(mes,dia):

    if((mes==1) | (mes==2)):

        return 'Invierno'

    if(mes==3):

        if(dia<21):

            return 'Invierno'

        if(dia>=21):

            return 'Primavera'

    if((mes==4) | (mes==5)):

        return 'Primavera'

    if(mes==6):

        if(dia<21):

            return 'Primavera'

        if(dia>=21):

            return 'Verano'

    if((mes==7) | (mes==8)):

        return 'Verano'

    if(mes==9):

        if(dia<21):

            return 'Verano'

        if(dia>=21):

            return 'Otonio'

    if((mes==10) | (mes==11)):

        return 'Otonio'

    if(mes==12):

        if(dia<21):

            return 'Otonio'

        if(dia>=21):

            return 'Invierno'
#Aplicamos la función en cuestión

propiedades['Estacion']=propiedades.apply(lambda x: fechaEstacion(x['Mes'],x['Dia']),axis=1)

propiedades.head()
#Generamos una columna que tenga la diferencia entre los Metros Totales y los Curbiertos

propiedades['difmetros']=propiedades['metrostotales']-propiedades['metroscubiertos']

propiedades.head()
cantidad_prop_anio=propiedades['Anio'].value_counts().sort_index()

plt.subplots(figsize=(10,10))

grafico_anio=sns.barplot(x=cantidad_prop_anio.values,y=cantidad_prop_anio.index,orient='h',palette='magma')

grafico_anio.set_title("Cantidad de Propiedades en Venta por Año",fontsize=20)

grafico_anio.set_xlabel("Cantidad de Propiedades",fontsize=12)

grafico_anio.set_ylabel("Años",fontsize=12)
plot = propiedades.groupby(['Anio']).agg({'habitaciones':'median'}).plot(rot=0, linewidth=2,figsize=(12,8),legend=False,color='tomato');

plot.set_title('Evolucion de la Mediana de Habitaciones a lo largo de los Años', fontsize=20);

plot.set_xlabel('Año', fontsize=18)

plot.set_ylabel('Habitaciones', fontsize=18)

plt.xticks([2012,2013,2014,2015,2016])

plt.yticks([1,2,3,4,5,6,7,8,9,10])
plot = propiedades.groupby(['Anio']).agg({'gimnasio':'sum'}).plot(rot=0, linewidth=2,figsize=(12,8),legend=False,color='lime');

plot.set_title('Evolucion de la Cantidad de Propiedades con Gimnasio con el correr de los Años', fontsize=18);

plot.set_xlabel('Año', fontsize=18)

plot.set_ylabel('Cantidad', fontsize=18)

plt.xticks([2012,2013,2014,2015,2016])
plot = propiedades.groupby(['Anio']).agg({'piscina':'sum'}).plot(rot=0, linewidth=2,figsize=(12,8),color='cornflowerblue',legend=False);

plot.set_title('Evolucion de la Cantidad de Propiedades con Piscinas con el correr de los Años', fontsize=18);

plot.set_xlabel('Año', fontsize=18)

plot.set_ylabel('Cantidad', fontsize=18)

plt.xticks([2012,2013,2014,2015,2016])
plot = propiedades.groupby(['Anio']).agg({'metrostotales':'mean'}).plot(rot=0, linewidth=2,figsize=(12,8),legend=False,color='saddlebrown');

plot.set_title('Evolucion del Promedio de Metros Totales a lo largo de los Años', fontsize=20);

plot.set_xlabel('Año', fontsize=18)

plot.set_ylabel('Metros Totales', fontsize=18)

plt.xticks([2012,2013,2014,2015,2016])
plot = propiedades.groupby(['Anio']).agg({'metroscubiertos':'mean'}).plot(rot=0, linewidth=2,figsize=(12,8),legend=False,color='chocolate');

plot.set_title('Evolucion del Promedio de Metros Cubiertos a lo largo de los Años', fontsize=20);

plot.set_xlabel('Año', fontsize=18)

plot.set_ylabel('Metros Cubiertos', fontsize=18)

plt.xticks([2012,2013,2014,2015,2016])
cantidad_prop_mes=propiedades['Mes'].value_counts().sort_index().reset_index()

cantidad_prop_mes['index'].replace({1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'},inplace=True)

cantidad_prop_mes.set_index(['index'],inplace=True)

grafico_mes=cantidad_prop_mes.plot(kind='bar',color='orchid',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_mes.set_title("Cantidad de Propiedades en Venta por Mes",fontsize=20)

grafico_mes.set_xlabel("Mes",fontsize=12)

grafico_mes.set_ylabel("Cantidad de Propiedades",fontsize=12)
cantidad_prop_dia=propiedades['Dia'].value_counts().sort_index().head(30)

grafico_dia=cantidad_prop_dia.plot(kind='bar',color='royalblue',fontsize=12,figsize=(10,10),rot=0)

grafico_dia.set_title("Cantidad de Propiedades en Venta por Día",fontsize=20)

grafico_dia.set_xlabel("Día",fontsize=12)

grafico_dia.set_ylabel("Cantidad de Propiedades",fontsize=12)
fecha_propiedades=propiedades[['Anio','Mes','Dia']].copy()

fecha_propiedades['Valor']=1

anio_mes=fecha_propiedades.pivot_table(index='Mes',columns='Anio',values='Valor',aggfunc='count')

anio_mes.rename(index={1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'},inplace=True)

anio_mes
plt.subplots(figsize=(10,10))

grafico_anio_mes=sns.heatmap(anio_mes,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")

grafico_anio_mes.set_title("Cantidad de Propiedades en Venta por Mes según Año",fontsize=20)

grafico_anio_mes.set_xlabel("Año",fontsize=12)

grafico_anio_mes.set_ylabel("Mes",fontsize=12)

grafico_anio_mes.set_yticklabels(grafico_anio_mes.get_yticklabels(),rotation=0)
mes_dia=fecha_propiedades.pivot_table(index='Mes',columns='Dia',values='Valor',aggfunc='count')

mes_dia.fillna(0,inplace=True)

mes_dia.drop(columns={31},inplace=True)

mes_dia.rename(index={1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'},inplace=True)

mes_dia
plt.subplots(figsize=(12,12))

grafico_mes_dia=sns.heatmap(mes_dia,linewidths=.5,cmap="nipy_spectral_r")

grafico_mes_dia.set_title("Cantidad de Propiedades en Venta por Dia según Mes",fontsize=20)

grafico_mes_dia.set_xlabel("Dia",fontsize=12)

grafico_mes_dia.set_ylabel("Mes",fontsize=12)

grafico_mes_dia.set_yticklabels(grafico_mes_dia.get_yticklabels(),rotation=0)
precio_anio_mes=propiedades[['Anio','Mes','precio']].copy()

group_precio_anio_mes=precio_anio_mes.pivot_table(index='Mes',columns='Anio',values='precio',aggfunc='mean')

group_precio_anio_mes
grafico_group_precio_anio_mes=group_precio_anio_mes.plot(kind='line',color=['royalblue','black','tomato','chartreuse','orchid'],figsize=(10,10),fontsize=12)

grafico_group_precio_anio_mes.set_title("Precio Promedio de Propiedades por Año según Mes",fontsize=20)

grafico_group_precio_anio_mes.set_xlabel("Mes",fontsize=12)

grafico_group_precio_anio_mes.set_ylabel("Precio",fontsize=12)

leyenda=plt.legend(['2012','2013','2014','2015','2016'],fontsize=10,title='Año',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
tipo_propiedad_top=propiedades['tipodepropiedad'].value_counts().head(10).reset_index()

tipo_propiedad_top.drop(columns={'tipodepropiedad'},inplace=True)

tipo_propiedad_top.rename(columns={'index':'tipodepropiedad'},inplace=True)

tipo_propiedad_top
año_tipo=propiedades[['Anio','tipodepropiedad']].copy()

año_tipo.dropna(inplace=True)

año_tipo_top=pd.merge(año_tipo,tipo_propiedad_top,on='tipodepropiedad',how='inner')

año_tipo_top['Valor']=1

cantidad_año_tipo_top=año_tipo_top.pivot_table(index='tipodepropiedad',columns='Anio',values='Valor',aggfunc='sum')

cantidad_año_tipo_top.head()
plt.subplots(figsize=(10,10))

grafico_cantidad_año_tipo_top=sns.heatmap(cantidad_año_tipo_top,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")

grafico_cantidad_año_tipo_top.set_title("Cantidad de Tipo de Propiedades por Año",fontsize=20)

grafico_cantidad_año_tipo_top.set_xlabel("Año",fontsize=12)

grafico_cantidad_año_tipo_top.set_ylabel("Tipo de Propiedad",fontsize=12)

grafico_cantidad_año_tipo_top.set_yticklabels(grafico_cantidad_año_tipo_top.get_yticklabels(),rotation=0)
cantidad_propiedades=propiedades['provincia'].value_counts().reset_index()

cantidad_propiedades.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_propiedades
cantidad_propiedades_provicia=pd.merge(mex,cantidad_propiedades,on='provincia',how='left')

cantidad_propiedades_provicia.fillna(0,inplace=True)

cantidad_propiedades_provicia.head()
fig, grafico_cantidad_propiedades_provicia = plt.subplots(1, figsize=(20, 10))

test=cantidad_propiedades_provicia.plot(column='cantidad',cmap='BuGn', edgecolor='black',ax=grafico_cantidad_propiedades_provicia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_propiedades_provicia.axis('off')

grafico_cantidad_propiedades_provicia.set_title("Cantidad de Propiedades en Venta por Estado",fontsize=22)
precio_provincia=propiedades.groupby(['provincia']).agg({'precio':'mean'})

precio_provincia=precio_provincia.reset_index()

precio_provincia
precio_provincia_mex=pd.merge(mex,precio_provincia,on='provincia',how='left')

precio_provincia_mex.fillna(0,inplace=True)

precio_provincia_mex.head()
fig, grafico_precio_provincia_mex = plt.subplots(1, figsize=(20, 10))

precio_provincia_mex.plot(column='precio',cmap='YlOrBr', edgecolor='black',ax=grafico_precio_provincia_mex,legend=True,figsize=(30,30),linewidth=0.5)

grafico_precio_provincia_mex.axis('off')

grafico_precio_provincia_mex.set_title("Precio Promedio por Estado",fontsize=22)
cantidad_propiedades.set_index(['provincia'],inplace=True)

cantidad_propiedades.sort_index(inplace=True)

cantidad_propiedades.reset_index(inplace=True)
cantidad_propiedades_precio=pd.merge(cantidad_propiedades,precio_provincia,on='provincia',how='inner')

cantidad_propiedades_precio['precio_propiedad']=cantidad_propiedades_precio['precio']/cantidad_propiedades_precio['cantidad']

cantidad_propiedades_precio.drop(columns={'cantidad','precio'},inplace=True)

cantidad_propiedades_precio
precio_propiedad_provincia_mex=pd.merge(mex,cantidad_propiedades_precio,on='provincia',how='left')

precio_propiedad_provincia_mex.fillna(0,inplace=True)

precio_propiedad_provincia_mex.head()
fig, grafico_precio_propiedad_provincia_mex= plt.subplots(1, figsize=(20, 10))

precio_propiedad_provincia_mex.plot(column='precio_propiedad',cmap='pink', edgecolor='black',ax=grafico_precio_propiedad_provincia_mex,legend=True,figsize=(30,30),linewidth=0.5)

grafico_precio_propiedad_provincia_mex.axis('off')

grafico_precio_propiedad_provincia_mex.set_title("Precio Promedio de Propiedades por Estado",fontsize=22)
casa=propiedades[(propiedades['tipodepropiedad']=='Casa')]

cantidad_casa_provincia=casa['provincia'].value_counts().reset_index()

cantidad_casa_provincia.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_casa_provincia
cantidad_casa_provincia_mex=pd.merge(mex,cantidad_casa_provincia,on='provincia',how='left')

cantidad_casa_provincia_mex.fillna(0,inplace=True)

cantidad_casa_provincia_mex.head()
fig, grafico_cantidad_casa_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_casa_provincia_mex.plot(column='cantidad',cmap='BuPu', edgecolor='black',ax=grafico_cantidad_casa_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_casa_provincia.axis('off')

grafico_cantidad_casa_provincia.set_title("Cantidad de Casas en Venta por Estado",fontsize=20)
apartamento=propiedades[(propiedades['tipodepropiedad']=='Apartamento')]

cantidad_apartamento_provincia=apartamento['provincia'].value_counts().reset_index()

cantidad_apartamento_provincia.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_apartamento_provincia
cantidad_apartamento_provincia_mex=pd.merge(mex,cantidad_apartamento_provincia,on='provincia',how='left')

cantidad_apartamento_provincia_mex.fillna(0,inplace=True)

cantidad_apartamento_provincia_mex.head()
fig, grafico_cantidad_apartamento_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_apartamento_provincia_mex.plot(column='cantidad',cmap='Reds', edgecolor='black',ax=grafico_cantidad_apartamento_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_apartamento_provincia.axis('off')

grafico_cantidad_apartamento_provincia.set_title("Cantidad de Apartamentos en Venta por Estado",fontsize=20)
edificio=propiedades[(propiedades['tipodepropiedad']=='Edificio')]

cantidad_edificio_provincia=edificio['provincia'].value_counts().reset_index()

cantidad_edificio_provincia.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_edificio_provincia
cantidad_edificio_provincia_mex=pd.merge(mex,cantidad_edificio_provincia,on='provincia',how='left')

cantidad_edificio_provincia_mex.fillna(0,inplace=True)

cantidad_edificio_provincia_mex.head()
fig, grafico_cantidad_edificio_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_edificio_provincia_mex.plot(column='cantidad',cmap='tab20c_r', edgecolor='black',ax=grafico_cantidad_edificio_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_edificio_provincia.axis('off')

grafico_cantidad_edificio_provincia.set_title("Cantidad de Edificios en Venta por Estado",fontsize=20)
terreno=propiedades[(propiedades['tipodepropiedad']=='Terreno')]

cantidad_terreno_provincia=terreno['provincia'].value_counts().reset_index()

cantidad_terreno_provincia.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_terreno_provincia
cantidad_terreno_provincia_mex=pd.merge(mex,cantidad_terreno_provincia,on='provincia',how='left')

cantidad_terreno_provincia_mex.fillna(0,inplace=True)

cantidad_terreno_provincia_mex.head()
fig, grafico_cantidad_terreno_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_terreno_provincia_mex.plot(column='cantidad',cmap='Greys', edgecolor='black',ax=grafico_cantidad_terreno_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_terreno_provincia.axis('off')

grafico_cantidad_terreno_provincia.set_title("Cantidad de Terrenos en Venta por Estado",fontsize=20)
local_comercial=propiedades[(propiedades['tipodepropiedad']=='Local Comercial')]

cantidad_local_comercial=local_comercial['provincia'].value_counts().reset_index()

cantidad_local_comercial.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_local_comercial
cantidad_local_comercial_provincia_mex=pd.merge(mex,cantidad_local_comercial,on='provincia',how='left')

cantidad_local_comercial_provincia_mex.fillna(0,inplace=True)

cantidad_local_comercial_provincia_mex.head()
fig, grafico_cantidad_local_comercial_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_local_comercial_provincia_mex.plot(column='cantidad',cmap='Greens', edgecolor='black',ax=grafico_cantidad_local_comercial_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_local_comercial_provincia.axis('off')

grafico_cantidad_local_comercial_provincia.set_title("Cantidad de Locales Comerciales en Venta por Estado",fontsize=20)
oficina_comercial=propiedades[(propiedades['tipodepropiedad']=='Oficina comercial')]

cantidad_oficina_comercial=oficina_comercial['provincia'].value_counts().reset_index()

cantidad_oficina_comercial.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_oficina_comercial
cantidad_oficina_comercial_provincia_mex=pd.merge(mex,cantidad_oficina_comercial,on='provincia',how='left')

cantidad_oficina_comercial_provincia_mex.fillna(0,inplace=True)

cantidad_oficina_comercial_provincia_mex.head()
fig, grafico_cantidad_oficina_comercial_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_oficina_comercial_provincia_mex.plot(column='cantidad',cmap='Oranges', edgecolor='black',ax=grafico_cantidad_oficina_comercial_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_oficina_comercial_provincia.axis('off')

grafico_cantidad_oficina_comercial_provincia.set_title("Cantidad de Oficinas Comerciales en Venta por Estado",fontsize=20)
duplex=propiedades[(propiedades['tipodepropiedad']=='Duplex')]

cantidad_duplex=duplex['provincia'].value_counts().reset_index()

cantidad_duplex.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_duplex
cantidad_duplex_provincia_mex=pd.merge(mex,cantidad_duplex,on='provincia',how='left')

cantidad_duplex_provincia_mex.fillna(0,inplace=True)

cantidad_duplex_provincia_mex.head()
fig, grafico_cantidad_duplex_provincia = plt.subplots(1, figsize=(20, 10))

cantidad_duplex_provincia_mex.plot(column='cantidad',cmap='Purples', edgecolor='black',ax=grafico_cantidad_duplex_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_cantidad_duplex_provincia.axis('off')

grafico_cantidad_duplex_provincia.set_title("Cantidad de Duplex en Venta por Estado",fontsize=20)
antiguedad=propiedades[['provincia','antiguedad']].copy()

antiguedad.dropna(inplace=True)

antiguedad_provincia=antiguedad.groupby(['provincia']).agg({'antiguedad':'mean'})

antiguedad_provincia.reset_index(inplace=True)

antiguedad_provincia
antiguedad_provincias_mex=pd.merge(mex,antiguedad_provincia,on='provincia',how='left')

antiguedad_provincias_mex.fillna(0,inplace=True)

antiguedad_provincias_mex.head()
fig, grafico_antiguedad_provincias = plt.subplots(1, figsize=(20, 10))

antiguedad_provincias_mex.plot(column='antiguedad',cmap='tab20', edgecolor='black',ax=grafico_antiguedad_provincias,legend=True,figsize=(30,30),linewidth=0.5)

grafico_antiguedad_provincias.axis('off')

grafico_antiguedad_provincias.set_title("Promedio de Antigüedad de las Propiedades por Estado",fontsize=22)
#Contamos la cantidad de cada uno

cant_sum=propiedades['usosmultiples'].value_counts().sort_index().reset_index()

cant_gim=propiedades['gimnasio'].value_counts().sort_index().reset_index()

cant_piscina=propiedades['piscina'].value_counts().sort_index().reset_index()

cant_escuelas=propiedades['escuelascercanas'].value_counts().sort_index().reset_index()

cant_centros=propiedades['centroscomercialescercanos'].value_counts().sort_index().reset_index()
#Los juntmos todos en un solo dataframe

cantidad_total=pd.merge(cant_sum,cant_gim,on='index',how='inner')

cantidad_total=pd.merge(cantidad_total,cant_piscina,on='index',how='inner')

cantidad_total=pd.merge(cantidad_total,cant_escuelas,on='index',how='inner')

cantidad_total=pd.merge(cantidad_total,cant_centros,on='index',how='inner')
#Renombramos los index y reemplazamos los valores para que tengan sentido

cantidad_total['index'].replace({0.0:'NO',1.0:'SI'},inplace=True)

cantidad_total.rename(columns={'index':'Hay','usosmultiples':'SUM','gimnasio':'Gimnasio','piscina':'Piscina','escuelascercanas':'Escuelas','centroscomercialescercanos':'Centros'},inplace=True)

cantidad_total.set_index('Hay',inplace=True)

cantidad_total
grafico_cantidad_total=cantidad_total.plot(kind='bar',cmap='Set1',figsize=(10,10),fontsize=16,rot=0)

grafico_cantidad_total.set_title("Cantidad según hay o no hay",fontsize=20)

grafico_cantidad_total.set_xlabel("¿Hay?",fontsize=14)

grafico_cantidad_total.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(fontsize=16,title='Tipo',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
#Filtamos las coordenadas para que solamente esten las que se encuentren en México y solamente las que tengan piscina

piscina=propiedades[['lat','lng','piscina']].copy()

piscina.dropna(inplace=True)

piscina_cerca=piscina[(piscina['piscina']==1.0)]

filtrado_piscina_cerca=piscina_cerca[(piscina_cerca['lat']<=lat_maxima) & (piscina_cerca['lat']>=lat_minima) & (piscina_cerca['lng']<=lng_maxima) & (piscina_cerca['lng']>=lng_minima)]

filtrado_piscina_cerca.head()
#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(filtrado_piscina_cerca['lng'], filtrado_piscina_cerca['lat'])]

coordenadas_piscina_cerca = gpd.GeoDataFrame(filtrado_piscina_cerca, geometry=geometry)   

coordenadas_piscina_cerca.head()
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_piscinas=coordenadas_piscina_cerca.plot(column='piscina',cmap=ListedColormap('darkturquoise'),ax=ax,markersize=200)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_piscinas.set_title("Distribución de Propiedades con Piscina por Estado",fontsize=60)
#Filtamos las coordenadas para que solamente esten las que se encuentren en México y solamente las que tengan piscina

escuelas=propiedades[['lat','lng','escuelascercanas']].copy()

escuelas.dropna(inplace=True)

escuelas_cerca=escuelas[(escuelas['escuelascercanas']==1.0)]

filtrado_escuelas_cerca=escuelas_cerca[(escuelas_cerca['lat']<=lat_maxima) & (escuelas_cerca['lat']>=lat_minima) & (escuelas_cerca['lng']<=lng_maxima) & (escuelas_cerca['lng']>=lng_minima)]

filtrado_escuelas_cerca.head()
#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(filtrado_escuelas_cerca['lng'], filtrado_escuelas_cerca['lat'])]

coordenadas_escuelas_cerca = gpd.GeoDataFrame(filtrado_escuelas_cerca, geometry=geometry)   

coordenadas_escuelas_cerca.head()
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_escuelas=coordenadas_escuelas_cerca.plot(column='escuelascercanas',cmap=ListedColormap('tomato'),ax=ax,markersize=150)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_escuelas.set_title("Distribución de Propiedades con Escuelas Cercanas por Estado",fontsize=60)
#Filtamos las coordenadas para que solamente esten las que se encuentren en México y solamente las que tengan piscina

centros_comerciales=propiedades[['lat','lng','centroscomercialescercanos']].copy()

centros_comerciales.dropna(inplace=True)

centros_comerciales_cerca=centros_comerciales[(centros_comerciales['centroscomercialescercanos']==1.0)]

filtrado_centros_comerciales_cerca=centros_comerciales_cerca[(centros_comerciales_cerca['lat']<=lat_maxima) & (centros_comerciales_cerca['lat']>=lat_minima) & (centros_comerciales_cerca['lng']<=lng_maxima) & (centros_comerciales_cerca['lng']>=lng_minima)]

filtrado_centros_comerciales_cerca.head()
#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(filtrado_centros_comerciales_cerca['lng'], filtrado_centros_comerciales_cerca['lat'])]

coordenadas_centros_comerciales_cerca = gpd.GeoDataFrame(filtrado_centros_comerciales_cerca, geometry=geometry)   

coordenadas_centros_comerciales_cerca.head()
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_centros_comerciales=coordenadas_centros_comerciales_cerca.plot(column='centroscomercialescercanos',cmap=ListedColormap('lime'),ax=ax,markersize=150)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_centros_comerciales.set_title("Distribución de Propiedades con Centros Comerciales Cercanos por Estado",fontsize=60)
#Filtamos las coordenadas para que solamente esten las que se encuentren en México y solamente las que tengan piscina

gimnasio=propiedades[['lat','lng','gimnasio']].copy()

gimnasio.dropna(inplace=True)

gimnasio_cerca=gimnasio[(gimnasio['gimnasio']==1)]

filtrado_gimnasio_cerca=gimnasio_cerca[(gimnasio_cerca['lat']<=lat_maxima) & (gimnasio_cerca['lat']>=lat_minima) & (gimnasio_cerca['lng']<=lng_maxima) & (gimnasio_cerca['lng']>=lng_minima)]

filtrado_gimnasio_cerca.head()
#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(filtrado_gimnasio_cerca['lng'], filtrado_gimnasio_cerca['lat'])]

coordenadas_gimnasio_cerca = gpd.GeoDataFrame(filtrado_gimnasio_cerca, geometry=geometry)   

coordenadas_gimnasio_cerca.head()
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_gimnasio=coordenadas_gimnasio_cerca.plot(column='gimnasio',cmap=ListedColormap('cornflowerblue'),ax=ax,markersize=200)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_gimnasio.set_title("Distribución de Propiedades con Gimnasio por Estado",fontsize=60)
#Filtamos las coordenadas para que solamente esten las que se encuentren en México y solamente las que tengan piscina

SUM=propiedades[['lat','lng','usosmultiples']].copy()

SUM.dropna(inplace=True)

SUM_cerca=SUM[(SUM['usosmultiples']==1)]

filtrado_SUM_cerca=SUM_cerca[(SUM_cerca['lat']<=lat_maxima) & (SUM_cerca['lat']>=lat_minima) & (SUM_cerca['lng']<=lng_maxima) & (SUM_cerca['lng']>=lng_minima)]

filtrado_SUM_cerca.head()
#Transformamos esas coordenadas a formato POINT con funcion Point

geometry = [Point(xy) for xy in zip(filtrado_SUM_cerca['lng'], filtrado_SUM_cerca['lat'])]

coordenadas_SUM_cerca = gpd.GeoDataFrame(filtrado_SUM_cerca, geometry=geometry)   

coordenadas_SUM_cerca.head()
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_SUM=coordenadas_SUM_cerca.plot(column='usosmultiples',cmap=ListedColormap('chocolate'),ax=ax,markersize=200)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_SUM.set_title("Distribución de Propiedades con SUM por Estado",fontsize=60)
ax=mex.plot(color='white',edgecolor='black',figsize=(40,40),linewidth=2.5)

ax.axis('off')

grafico_gimnasio=coordenadas_gimnasio_cerca.plot(column='gimnasio',cmap=ListedColormap('cornflowerblue'),ax=ax,markersize=300)

grafico_SUM=coordenadas_SUM_cerca.plot(column='usosmultiples',cmap=ListedColormap('chocolate'),ax=ax,markersize=150)

capitales_provincia.plot(cmap=ListedColormap('black'),ax=ax,markersize=500,marker='$\u29BF$',linewidth=1.5)

grafico_gimnasio.set_title("Distribución de Propiedades con Gimnasio y SUM por Estado",fontsize=60)

leyenda=plt.legend(['Gimnasio','SUM'],fontsize=50,frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.set_title("Tiene",prop=dict(size=50))

leyenda.legendHandles[0]._sizes = [700]

leyenda.legendHandles[1]._sizes = [700]

leyenda.get_frame().set_linewidth(4.0)
#Primero de todos los tipos en general

precio_tipo=propiedades.groupby(['tipodepropiedad']).agg({'precio':'mean'})

precio_tipo=precio_tipo.reset_index()

plt.subplots(figsize=(8,10))

grafico_precio_promedio_por_tipo=sns.barplot(y=precio_tipo['tipodepropiedad'],x=precio_tipo['precio'], orient='h')

grafico_precio_promedio_por_tipo.set_title("Precio Promedio por Tipo de Propiedad",fontsize=20)

grafico_precio_promedio_por_tipo.set_xlabel("Precio",fontsize=15)

grafico_precio_promedio_por_tipo.set_ylabel("Tipo de Propiedad",fontsize=15)
#Ahora solo nos quedamos con el precio de solamente los 10 primeros tipo

tipo_precio=propiedades[['tipodepropiedad','precio']].copy()

tipo_precio.dropna(subset=['tipodepropiedad'],inplace=True)

tipo_precio_top=pd.merge(tipo_precio,tipo_propiedad_top,on='tipodepropiedad',how='inner')

tipo_precio_top.head()
plt.subplots(figsize=(10,10))

grafico_tipo_precio_top=sns.boxplot(y=tipo_precio_top['precio'],x=tipo_precio_top['tipodepropiedad'])

grafico_tipo_precio_top.set_xticklabels(grafico_tipo_precio_top.get_xticklabels(),rotation=70)

grafico_tipo_precio_top.set_title("Precio de los 10 Tipos de Propiedades más populares",fontsize=18)

grafico_tipo_precio_top.set_ylabel("Precio",fontsize=14)

grafico_tipo_precio_top.set_xlabel("Tipo de Propiedad",fontsize=14)
antiguedad_precio=propiedades[['antiguedad','precio']].copy()

antiguedad_precio.dropna(inplace=True)

group_antiguedad_precio=antiguedad_precio.groupby(['antiguedad']).agg({'precio':'mean'})

group_antiguedad_precio.head()
grafico_antiguedad_precio=group_antiguedad_precio.plot(kind='line',figsize=(10,10),fontsize=12,legend=False)

grafico_antiguedad_precio.set_title("Precio promedio según Antigüedad",fontsize=20)

grafico_antiguedad_precio.set_xlabel("Antigüedad",fontsize=14)

grafico_antiguedad_precio.set_ylabel("Precio",fontsize=14)
habitaciones_precio=propiedades[['habitaciones','precio']].copy()

habitaciones_precio.dropna(inplace=True)

habitaciones_precio['habitaciones']=habitaciones_precio['habitaciones'].astype('int32')

group_habitaciones_precio=habitaciones_precio.groupby(['habitaciones']).agg({'precio':'mean'})

group_habitaciones_precio
plt.subplots(figsize=(10,10))

grafico_habitaciones_precio=sns.barplot(y=group_habitaciones_precio['precio'],x=group_habitaciones_precio.index)

grafico_habitaciones_precio.set_title("Precio Promedio por Cantidad de Habitaciones",fontsize=20)

grafico_habitaciones_precio.set_xlabel("Habitaciones",fontsize=14)

grafico_habitaciones_precio.set_ylabel("Precio",fontsize=14)
baños_precio=propiedades[['banos','precio']].copy()

baños_precio.dropna(inplace=True)

baños_precio['banos']=baños_precio['banos'].astype('int32')

group_baños_precio=baños_precio.groupby(['banos']).agg({'precio':'mean'})

group_baños_precio
plt.subplots(figsize=(10,10))

grafico_baños_precio=sns.barplot(x=group_baños_precio['precio'],y=group_baños_precio.index,orient='h',palette='plasma')

grafico_baños_precio.set_title("Precio Promedio por Cantidad de Baños",fontsize=20)

grafico_baños_precio.set_xlabel("Precio",fontsize=14)

grafico_baños_precio.set_ylabel("Baños",fontsize=14)
garages_precio=propiedades[['garages','precio']].copy()

garages_precio.dropna(inplace=True)

garages_precio['garages']=garages_precio['garages'].astype('int32')

group_garages_precio=garages_precio.groupby(['garages']).agg({'precio':'mean'})

group_garages_precio
plt.subplots(figsize=(10,10))

grafico_garages_precio=sns.barplot(x=group_garages_precio['precio'],y=group_garages_precio.index,orient='h',palette='hsv')

grafico_garages_precio.set_title("Precio Promedio por Cantidad de Garajes",fontsize=20)

grafico_garages_precio.set_xlabel("Precio",fontsize=14)

grafico_garages_precio.set_ylabel("Garajes",fontsize=14)
piscina_precio=propiedades[['piscina','precio']].copy()

piscina_precio.dropna(inplace=True)

piscina_precio['piscina'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_piscina_precio=piscina_precio.groupby(['piscina']).agg({'precio':'mean'})

group_piscina_precio
plt.subplots(figsize=(10,10))

grafico_piscina_precio=sns.barplot(x=group_piscina_precio['precio'],y=group_piscina_precio.index,orient='h',palette='rocket')

grafico_piscina_precio.set_title("Precio Promedio si la Propiedad tiene Piscina",fontsize=20)

grafico_piscina_precio.set_xlabel("Precio",fontsize=14)

grafico_piscina_precio.set_ylabel("¿Tiene Piscina?",fontsize=14)
escuelas_precio=propiedades[['escuelascercanas','precio']].copy()

escuelas_precio.dropna(inplace=True)

escuelas_precio['escuelascercanas'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_escuelas_precio=escuelas_precio.groupby(['escuelascercanas']).agg({'precio':'mean'})

group_escuelas_precio
plt.subplots(figsize=(10,10))

grafico_escuelas_precio=sns.barplot(x=group_escuelas_precio['precio'],y=group_escuelas_precio.index,orient='h',palette='gnuplot2')

grafico_escuelas_precio.set_title("Precio Promedio si la Propiedad tiene Escuelas Cercanas",fontsize=20)

grafico_escuelas_precio.set_xlabel("Precio",fontsize=14)

grafico_escuelas_precio.set_ylabel("¿Tiene Escuela Cercana?",fontsize=14)
gimnasio_precio=propiedades[['gimnasio','precio']].copy()

gimnasio_precio.dropna(inplace=True)

gimnasio_precio['gimnasio'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_gimnasio_precio=gimnasio_precio.groupby(['gimnasio']).agg({'precio':'mean'})

group_gimnasio_precio
plt.subplots(figsize=(10,10))

grafico_gimnasio_precio=sns.barplot(x=group_gimnasio_precio['precio'],y=group_gimnasio_precio.index,orient='h',palette='inferno')

grafico_gimnasio_precio.set_title("Precio Promedio si la Propiedad tiene Gimnasio",fontsize=20)

grafico_gimnasio_precio.set_xlabel("Precio",fontsize=14)

grafico_gimnasio_precio.set_ylabel("¿Tiene Gimnasio?",fontsize=14)
sum_precio=propiedades[['usosmultiples','precio']].copy()

sum_precio.dropna(inplace=True)

sum_precio['usosmultiples'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_sum_precio=sum_precio.groupby(['usosmultiples']).agg({'precio':'mean'})

group_sum_precio
plt.subplots(figsize=(10,10))

grafico_sum_precio=sns.barplot(x=group_sum_precio['precio'],y=group_sum_precio.index,orient='h',palette='afmhot')

grafico_sum_precio.set_title("Precio Promedio si la Propiedad tiene SUM",fontsize=20)

grafico_sum_precio.set_xlabel("Precio",fontsize=14)

grafico_sum_precio.set_ylabel("¿Tiene SUM?",fontsize=14)
centros_precio=propiedades[['centroscomercialescercanos','precio']].copy()

centros_precio.dropna(inplace=True)

centros_precio['centroscomercialescercanos'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_centros_precio=centros_precio.groupby(['centroscomercialescercanos']).agg({'precio':'mean'})

group_centros_precio
plt.subplots(figsize=(10,10))

grafico_centros_precio=sns.barplot(x=group_centros_precio['precio'],y=group_centros_precio.index,orient='h',palette='seismic')

grafico_centros_precio.set_title("Precio Promedio si la Propiedad tiene Centros Comerciales Cercanos",fontsize=20)

grafico_centros_precio.set_xlabel("Precio",fontsize=14)

grafico_centros_precio.set_ylabel("¿Tiene Centros Comerciales Cercanos?",fontsize=14)
plt.subplots(figsize=(10,10))

precio_provincia_mas_caro=propiedades.groupby(['provincia'], sort=False)['precio'].max().sort_values(ascending=False)[0:19]

provincia_mas_caro = sns.barplot(x=precio_provincia_mas_caro.values,y=precio_provincia_mas_caro.index,orient='h',palette='rainbow')

provincia_mas_caro.set_title('Precio Más Caro por Provincia', fontsize=20);

provincia_mas_caro.set_xlabel('Precio', fontsize=18)

provincia_mas_caro.set_ylabel('Provincia', fontsize=18)
plt.subplots(figsize=(10,10))

precio_ciudad_mas_caro=propiedades.groupby(['ciudad'], sort=False)['precio'].max().sort_values(ascending=False)[0:19]

ciudad_mas_caro = sns.barplot(x=precio_ciudad_mas_caro.values,y=precio_ciudad_mas_caro.index,orient='h',palette='magma')

ciudad_mas_caro.set_title('Precio Más Caro por Ciudad', fontsize=20);

ciudad_mas_caro.set_xlabel('Precio', fontsize=18)

ciudad_mas_caro.set_ylabel('Ciudad', fontsize=18)
estacion_tipo=propiedades[['tipodepropiedad','Estacion']].copy()

tipo_propiedad_primeros=tipo_propiedad_top.head(4)

tipo_propiedad_resto=tipo_propiedad_top.tail(6)

estacion_tipo_primero=pd.merge(estacion_tipo,tipo_propiedad_primeros,on='tipodepropiedad',how='inner')

estacion_tipo_primero['Valor']=1

pivot_estacion_tipo_primero=estacion_tipo_primero.pivot_table(index='tipodepropiedad',columns='Estacion',values='Valor',aggfunc='count')

pivot_estacion_tipo_primero
grafico_estaciones_tipo_primero=pivot_estacion_tipo_primero.plot(kind='bar',color=['grey','saddlebrown','lime','royalblue'],fontsize=17,figsize=(10,10),rot=0)

grafico_estaciones_tipo_primero.set_title("Cantidad de Propiedades en Venta por Estación por Tipo (TOP)",fontsize=17)

grafico_estaciones_tipo_primero.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_estaciones_tipo_primero.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(['Invierno','Otoño','Primavera','Verano'],fontsize=15,title='Estación',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
estacion_tipo_resto=pd.merge(estacion_tipo,tipo_propiedad_resto,on='tipodepropiedad',how='inner')

estacion_tipo_resto['Valor']=1

pivot_estacion_tipo_resto=estacion_tipo_resto.pivot_table(index='tipodepropiedad',columns='Estacion',values='Valor',aggfunc='count')

pivot_estacion_tipo_resto
grafico_estaciones_tipo_resto=pivot_estacion_tipo_resto.plot(kind='bar',color=['grey','saddlebrown','lime','royalblue'],fontsize=17,figsize=(10,10),rot=50)

grafico_estaciones_tipo_resto.set_title("Cantidad de Propiedades en Venta por Estación por Tipo(Resto)",fontsize=17)

grafico_estaciones_tipo_resto.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_estaciones_tipo_resto.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(['Invierno','Otoño','Primavera','Verano'],fontsize=15,title='Estación',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
group_precio_estacion=propiedades.groupby(['Estacion']).agg({'precio':'mean'})

group_precio_estacion
plt.subplots(figsize=(10,10))

grafico_estaciones_precio=sns.barplot(x=group_precio_estacion.index,y=group_precio_estacion['precio'],palette=['grey','saddlebrown','lime','royalblue'])

grafico_estaciones_precio.set_title("Precio por Estación",fontsize=17)

grafico_estaciones_precio.set_xlabel("Estación",fontsize=14)

grafico_estaciones_precio.set_ylabel("Precio",fontsize=14)
precio_provincia=propiedades.groupby(['provincia','Estacion']).agg({'precio':'mean'})

precio_provincia=precio_provincia.reset_index()

precio_provincia_mex=pd.merge(mex,precio_provincia,on='provincia',how='inner')

precio_promedio_por_provincia_por_estacion = precio_provincia_mex.pivot("provincia", "Estacion", "precio")

precio_promedio_por_provincia_por_estacion.head()
plt.figure(figsize = (10,10))

ax = sns.heatmap(precio_promedio_por_provincia_por_estacion, linewidths=.5)

ax.set_title("Precio Promedio por Provincia por Estación", fontsize =20)

ax.set_ylabel("Provincia",fontsize=18)

ax.set_xlabel("Estación",fontsize=18)
#Creamos los labels necesarios para la Leyenda 

AL = mpatches.Patch(color='white',label='00 - Azcapotzalco')

GM = mpatches.Patch(color='white',label='01 - Gustavo A. Madero')

MH = mpatches.Patch(color='white',label='02 - Miguel Hidalgo')

IZ = mpatches.Patch(color='white',label='03 - Iztacalco')

VC = mpatches.Patch(color='white',label='04 - Venustiano Carranza')

IP = mpatches.Patch(color='white',label='05 - Iztapalapa')

TL = mpatches.Patch(color='white',label='06 - Tlalpan')

XO = mpatches.Patch(color='white',label='07 - Xochimilco')

MC = mpatches.Patch(color='white',label='08 - La Magdalena Contreras')

BJ = mpatches.Patch(color='white',label='09 - Benito Juárez')

CT = mpatches.Patch(color='white',label='10 - Cuauhtémoc')

AO = mpatches.Patch(color='white',label='11 - Álvaro Obregón')

CM = mpatches.Patch(color='white',label='12 - Cuajimalpa de Morelos')

CY = mpatches.Patch(color='white',label='13 - Coyoacán')

TH = mpatches.Patch(color='white',label='14 - Tláhuac')

MA = mpatches.Patch(color='white',label='15 - Milpa Alta')
#Nos quedamos con los municipios del Distrito Federal

muni_df=mex_municipio[(mex_municipio['provincia']=='09')]

muni_df.head()
df=propiedades[(propiedades['provincia']=='Distrito Federal')]

distrito=df['ciudad'].value_counts().head(16).reset_index()

distrito.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

distrito['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

muni_df_cantidad=pd.merge(muni_df,distrito,on='ciudad',how='left')

muni_df_cantidad.fillna(0,inplace=True)

muni_df_cantidad.head()
fig, grafico_distrito_ciudad = plt.subplots(1, figsize=(25, 20))

grafico_cantidad=muni_df_cantidad.plot(column='cantidad',cmap='Greens_r', edgecolor='black',ax=grafico_distrito_ciudad,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_cantidad,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_distrito_ciudad.axis('off')

grafico_distrito_ciudad.set_title("Cantidad de Propiedades en Venta por Municipio",fontsize=30)

for punto in muni_df_cantidad.iterrows():

    grafico_cantidad.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)    
precio_ciudad=df.groupby(['ciudad']).agg({'precio':'mean'})

precio_ciudad=precio_ciudad.reset_index().head(16)

precio_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

muni_df_precio=pd.merge(muni_df,precio_ciudad,on='ciudad',how='left')

muni_df_precio.fillna(0,inplace=True)

muni_df_precio.head()
fig, grafico_distrito_ciudad_precio = plt.subplots(1, figsize=(25, 20))

grafico_precio=muni_df_precio.plot(column='precio',cmap='ocean', edgecolor='black',ax=grafico_distrito_ciudad_precio,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_precio,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_distrito_ciudad_precio.axis('off')

grafico_distrito_ciudad_precio.set_title("Precio Promedio por Municipio",fontsize=30)

for punto in muni_df_precio.iterrows():

    grafico_precio.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)
df_casa=df[(df['tipodepropiedad']=='Casa')]

df_casa_ciudad=df_casa['ciudad'].value_counts().reset_index()

df_casa_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_casa_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_casa_muni=pd.merge(muni_df,df_casa_ciudad,on='ciudad',how='left')

df_casa_muni.fillna(0,inplace=True)

df_casa_muni.head()
fig, grafico_casa_distrito = plt.subplots(1, figsize=(25, 20))

grafico_casa=df_casa_muni.plot(column='cantidad',cmap='plasma', edgecolor='black',ax=grafico_casa_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_casa,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_casa_distrito.axis('off')

grafico_casa_distrito.set_title("Cantidad de Casas en Venta por Municipio",fontsize=30)

for punto in df_casa_muni.iterrows():

    grafico_casa.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)    
df_apartamento=df[(df['tipodepropiedad']=='Apartamento')]

df_apartamento_ciudad=df_apartamento['ciudad'].value_counts().reset_index()

df_apartamento_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_apartamento_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_apartamento_muni=pd.merge(muni_df,df_apartamento_ciudad,on='ciudad',how='left')

df_apartamento_muni.fillna(0,inplace=True)

df_apartamento_muni.head()
fig, grafico_apartamento_distrito = plt.subplots(1, figsize=(25, 20))

grafico_apartamento=df_apartamento_muni.plot(column='cantidad',cmap='Reds', edgecolor='black',ax=grafico_apartamento_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_apartamento,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_apartamento_distrito.axis('off')

grafico_apartamento_distrito.set_title("Cantidad de Apartamentos en Venta por Municipio",fontsize=30)

for punto in df_apartamento_muni.iterrows():

    grafico_apartamento.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)    
df_edificio=df[(df['tipodepropiedad']=='Edificio')]

df_edificio_ciudad=df_edificio['ciudad'].value_counts().reset_index()

df_edificio_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_edificio_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_edificio_muni=pd.merge(muni_df,df_edificio_ciudad,on='ciudad',how='left')

df_edificio_muni.fillna(0,inplace=True)

df_edificio_muni.head()
fig, grafico_edificio_distrito = plt.subplots(1, figsize=(25, 20))

grafico_edificio=df_edificio_muni.plot(column='cantidad',cmap='YlGn', edgecolor='black',ax=grafico_edificio_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_edificio,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_edificio_distrito.axis('off')

grafico_edificio_distrito.set_title("Cantidad de Edificios en Venta por Municipio",fontsize=30)

for punto in df_edificio_muni.iterrows():

    grafico_edificio.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)    
df_terreno=df[(df['tipodepropiedad']=='Terreno')]

df_terreno_ciudad=df_terreno['ciudad'].value_counts().reset_index()

df_terreno_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_terreno_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_terreno_muni=pd.merge(muni_df,df_terreno_ciudad,on='ciudad',how='left')

df_terreno_muni.fillna(0,inplace=True)

df_terreno_muni.head()
fig, grafico_terreno_distrito = plt.subplots(1, figsize=(25, 20))

grafico_terreno=df_terreno_muni.plot(column='cantidad',cmap='magma', edgecolor='black',ax=grafico_terreno_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_terreno,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_terreno_distrito.axis('off')

grafico_terreno_distrito.set_title("Cantidad de Terrenos en Venta por Municipio",fontsize=30)

for punto in df_terreno_muni.iterrows():

    grafico_terreno.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)   
df_locales=df[(df['tipodepropiedad']=='Local Comercial')]

df_locales_ciudad=df_locales['ciudad'].value_counts().reset_index()

df_locales_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_locales_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_locales_muni=pd.merge(muni_df,df_locales_ciudad,on='ciudad',how='left')

df_locales_muni.fillna(0,inplace=True)

df_locales_muni.head()
fig, grafico_locales_distrito = plt.subplots(1, figsize=(25, 20))

grafico_locales=df_locales_muni.plot(column='cantidad',cmap='Purples', edgecolor='black',ax=grafico_locales_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_locales,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_locales_distrito.axis('off')

grafico_locales_distrito.set_title("Cantidad de Locales Comerciales en Venta por Municipio",fontsize=30)

for punto in df_locales_muni.iterrows():

    grafico_locales.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)   
df_oficina=df[(df['tipodepropiedad']=='Oficina comercial')]

df_oficina_ciudad=df_oficina['ciudad'].value_counts().reset_index()

df_oficina_ciudad.rename(columns={'index':'ciudad','ciudad':'cantidad'},inplace=True)

df_oficina_ciudad['ciudad'].replace({'Alvaro Obregón':'Álvaro Obregón'},inplace=True)

df_oficina_muni=pd.merge(muni_df,df_oficina_ciudad,on='ciudad',how='left')

df_oficina_muni.fillna(0,inplace=True)

df_oficina_muni.head()
fig, grafico_oficina_distrito = plt.subplots(1, figsize=(25, 20))

grafico_oficina=df_oficina_muni.plot(column='cantidad',cmap='PuBuGn', edgecolor='black',ax=grafico_oficina_distrito,legend=True,figsize=(30,30),linewidth=1.0)

ciudad_mexico.plot(cmap=ListedColormap('black'),ax=grafico_oficina,markersize=300,marker='$\u29BF$',linewidth=1.5)

grafico_oficina_distrito.axis('off')

grafico_oficina_distrito.set_title("Cantidad de Oficinas Comerciales en Venta por Municipio",fontsize=30)

for punto in df_oficina_muni.iterrows():

    grafico_oficina.text(punto[1]['geometry'].centroid.x,punto[1]['geometry'].centroid.y,punto[0],horizontalalignment='center',fontsize=16,bbox=dict(boxstyle='round', facecolor='linen', alpha=1))

leyenda=plt.legend(handles=[AL,GM,MH,IZ,VC,IP,TL,XO,MC,BJ,CT,AO,CM,CY,TH,MA],loc=2,fontsize=14,frameon=True,facecolor='white',edgecolor='black') 

leyenda.set_title("Municipios",prop=dict(size=15))

leyenda.get_frame().set_linewidth(2.0)   
#Tomamos solamente los cuales la diferencia entre metros es mayor a 0

filtrado_metros=propiedades[(propiedades['difmetros']>=0.0)]

metros_totales=filtrado_metros.groupby(['provincia']).agg({'metrostotales':'mean'})

metros_totales.reset_index(inplace=True)

metros_totales
metros_totales_mex=pd.merge(mex,metros_totales,on='provincia',how='left')

metros_totales_mex.fillna(0,inplace=True)

metros_totales_mex.head()
fig, grafico_metros_totales_provincia = plt.subplots(1, figsize=(20, 10))

metros_totales_mex.plot(column='metrostotales',cmap='GnBu', edgecolor='black',ax=grafico_metros_totales_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_metros_totales_provincia.axis('off')

grafico_metros_totales_provincia.set_title("Promedio de Metros Totales por Estado",fontsize=20)
habitaciones_metros_totales=filtrado_metros[['habitaciones','metrostotales']].copy()

habitaciones_metros_totales.dropna(inplace=True)

habitaciones_metros_totales['habitaciones']=habitaciones_metros_totales['habitaciones'].astype('int32')

group_habitaciones_metros_totales=habitaciones_metros_totales.groupby(['habitaciones']).agg({'metrostotales':'mean'})

group_habitaciones_metros_totales
grafico_habitaciones_metros_totales=group_habitaciones_metros_totales.plot(kind='bar',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_habitaciones_metros_totales.set_title("Metros Totales Promedio según cantidad de Habitaciones",fontsize=20)

grafico_habitaciones_metros_totales.set_xlabel("Habitaciones",fontsize=14)

grafico_habitaciones_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
antiguedad_metros_totales=filtrado_metros[['antiguedad','metrostotales']].copy()

antiguedad_metros_totales.dropna(inplace=True)

group_antiguedad_metros_totales=antiguedad_metros_totales.groupby(['antiguedad']).agg({'metrostotales':'mean'})

group_antiguedad_metros_totales.head()
grafico_antiguedad_metros_totales=group_antiguedad_metros_totales.plot(kind='line',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_antiguedad_metros_totales.set_title("Metros Totales Promedio según la Antigüedad",fontsize=20)

grafico_antiguedad_metros_totales.set_xlabel("Antigüedad",fontsize=14)

grafico_antiguedad_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
baños_metros_totales=filtrado_metros[['banos','metrostotales']].copy()

baños_metros_totales.dropna(inplace=True)

baños_metros_totales['banos']=baños_metros_totales['banos'].astype('int32')

group_baños_metros_totales=baños_metros_totales.groupby(['banos']).agg({'metrostotales':'mean'})

group_baños_metros_totales
grafico_baños_metros_totales=group_baños_metros_totales.plot(kind='bar',color='tomato',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_baños_metros_totales.set_title("Metros Totales Promedio según cantidad de Baños",fontsize=20)

grafico_baños_metros_totales.set_xlabel("Baños",fontsize=14)

grafico_baños_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
garage_metros_totales=filtrado_metros[['garages','metrostotales']].copy()

garage_metros_totales.dropna(inplace=True)

garage_metros_totales['garages']=garage_metros_totales['garages'].astype('int32')

group_garage_metros_totales=garage_metros_totales.groupby(['garages']).agg({'metrostotales':'mean'})

group_garage_metros_totales
grafico_garage_metros_totales=group_garage_metros_totales.plot(kind='bar',color='purple',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_garage_metros_totales.set_title("Metros Totales Promedio según cantidad de Garajes",fontsize=20)

grafico_garage_metros_totales.set_xlabel("Garajes",fontsize=14)

grafico_garage_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
gimnasio_metros_totales=filtrado_metros[['gimnasio','metrostotales']].copy()

gimnasio_metros_totales.dropna(inplace=True)

gimnasio_metros_totales['gimnasio'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_gimnasio_metros_totales=gimnasio_metros_totales.groupby(['gimnasio']).agg({'metrostotales':'mean'})

group_gimnasio_metros_totales
grafico_gimnasio_metros_totales=group_gimnasio_metros_totales.plot(kind='bar',color='chocolate',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_gimnasio_metros_totales.set_title("Metros Totales Promedio según si hay Gimnasio",fontsize=20)

grafico_gimnasio_metros_totales.set_xlabel("¿Tiene Gimansio?",fontsize=14)

grafico_gimnasio_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
piscina_metros_totales=filtrado_metros[['piscina','metrostotales']].copy()

piscina_metros_totales.dropna(inplace=True)

piscina_metros_totales['piscina'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_piscina_metros_totales=piscina_metros_totales.groupby(['piscina']).agg({'metrostotales':'mean'})

group_piscina_metros_totales
grafico_piscina_metros_totales=group_piscina_metros_totales.plot(kind='bar',color='darkturquoise',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_piscina_metros_totales.set_title("Metros Totales Promedio según si hay Piscina",fontsize=20)

grafico_piscina_metros_totales.set_xlabel("¿Tiene Piscina?",fontsize=14)

grafico_piscina_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
sum_metros_totales=filtrado_metros[['usosmultiples','metrostotales']].copy()

sum_metros_totales.dropna(inplace=True)

sum_metros_totales['usosmultiples'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_sum_metros_totales=sum_metros_totales.groupby(['usosmultiples']).agg({'metrostotales':'mean'})

group_sum_metros_totales
grafico_sum_metros_totales=group_sum_metros_totales.plot(kind='bar',color='greenyellow',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_sum_metros_totales.set_title("Metros Totales Promedio según si hay SUM",fontsize=20)

grafico_sum_metros_totales.set_xlabel("¿Tiene SUM?",fontsize=14)

grafico_sum_metros_totales.set_ylabel("Metros Totales Promedio",fontsize=14)
tipo_propiedad_metros=pd.merge(filtrado_metros,tipo_propiedad_top,on='tipodepropiedad',how='inner')

tipo_propiedad_totales=tipo_propiedad_metros.groupby(['tipodepropiedad']).agg({'metrostotales':'mean'})

tipo_propiedad_totales
plt.subplots(figsize=(10,10))

grafico_tipo_totales=sns.barplot(x=tipo_propiedad_totales['metrostotales'],y=tipo_propiedad_totales.index,orient='h',palette='rainbow')

grafico_tipo_totales.set_title("Promedio de Metros Totales por Top 10 de Tipo de Propiedad",fontsize=20)

grafico_tipo_totales.set_xlabel("Promedio de Metros Totales",fontsize=12)

grafico_tipo_totales.set_ylabel("Tipo de Propiedad",fontsize=12)
gimnasio_tipo_metros=propiedades[['tipodepropiedad','metrostotales','metroscubiertos','gimnasio']].copy()

gimnasio_tipo_metros.dropna(inplace=True)

gimnasio_tipo_metros['gimnasio'].replace({0.0:'NO',1.0:'SI'},inplace=True)

pivot_gimnasio_tipo_metrostotales=gimnasio_tipo_metros.pivot_table(index='tipodepropiedad',columns='gimnasio',values='metrostotales',aggfunc='mean')

pivot_gimnasio_tipo_metrostotales.dropna(inplace=True)

pivot_gimnasio_tipo_metrostotales
grafico_gimnasio_tipo_metrostotales=pivot_gimnasio_tipo_metrostotales.plot(kind='bar',color=['chocolate','royalblue'],rot=70,fontsize=12,figsize=(10,10))

grafico_gimnasio_tipo_metrostotales.set_title("Promedio de Metros Totales por Tipo de Propiedad según si tiene Gimnasio",fontsize=20)

grafico_gimnasio_tipo_metrostotales.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_gimnasio_tipo_metrostotales.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Gimnasio?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
piscina_tipo_metros=propiedades[['tipodepropiedad','metrostotales','metroscubiertos','piscina']].copy()

piscina_tipo_metros.dropna(inplace=True)

piscina_tipo_metros['piscina'].replace({0.0:'NO',1.0:'SI'},inplace=True)

pivot_piscina_tipo_metrostotales=piscina_tipo_metros.pivot_table(index='tipodepropiedad',columns='piscina',values='metrostotales',aggfunc='mean')

pivot_piscina_tipo_metrostotales.dropna(inplace=True)

pivot_piscina_tipo_metrostotales
grafico_piscina_tipo_metrostotales=pivot_piscina_tipo_metrostotales.plot(kind='bar',color=['tomato','darkturquoise'],rot=70,fontsize=12,figsize=(10,10))

grafico_piscina_tipo_metrostotales.set_title("Promedio de Metros Totales por Tipo de Propiedad según si tiene Piscina",fontsize=20)

grafico_piscina_tipo_metrostotales.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_piscina_tipo_metrostotales.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Piscina?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
sum_tipo_metros=propiedades[['tipodepropiedad','metrostotales','metroscubiertos','usosmultiples']].copy()

sum_tipo_metros.dropna(inplace=True)

sum_tipo_metros['usosmultiples'].replace({0.0:'NO',1.0:'SI'},inplace=True)

pivot_sum_tipo_metrostotales=sum_tipo_metros.pivot_table(index='tipodepropiedad',columns='usosmultiples',values='metrostotales',aggfunc='mean')

pivot_sum_tipo_metrostotales.dropna(inplace=True)

pivot_sum_tipo_metrostotales
grafico_sum_tipo_metrostotales=pivot_sum_tipo_metrostotales.plot(kind='bar',color=['saddlebrown','green'],rot=70,fontsize=12,figsize=(10,10))

grafico_sum_tipo_metrostotales.set_title("Promedio de Metros Totales por Tipo de Propiedad según si tiene SUM",fontsize=20)

grafico_sum_tipo_metrostotales.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_sum_tipo_metrostotales.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene SUM?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
metros_cubiertos=filtrado_metros.groupby(['provincia']).agg({'metroscubiertos':'mean'})

metros_cubiertos.reset_index(inplace=True)

metros_cubiertos
metros_cubiertos_mex=pd.merge(mex,metros_cubiertos,on='provincia',how='left')

metros_cubiertos_mex.fillna(0,inplace=True)

metros_cubiertos_mex.head()
fig, grafico_metros_cubiertos_provincia = plt.subplots(1, figsize=(20, 10))

metros_cubiertos_mex.plot(column='metroscubiertos',cmap='OrRd', edgecolor='black',ax=grafico_metros_cubiertos_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_metros_cubiertos_provincia.axis('off')

grafico_metros_cubiertos_provincia.set_title("Promedio de Metros Cubiertos por Estado",fontsize=20)
habitaciones_metros_cubiertos=filtrado_metros[['habitaciones','metroscubiertos']].copy()

habitaciones_metros_cubiertos.dropna(inplace=True)

habitaciones_metros_cubiertos['habitaciones']=habitaciones_metros_totales['habitaciones'].astype('int32')

group_habitaciones_metros_cubiertos=habitaciones_metros_cubiertos.groupby(['habitaciones']).agg({'metroscubiertos':'mean'})

group_habitaciones_metros_cubiertos
grafico_habitaciones_metros_cubiertos=group_habitaciones_metros_cubiertos.plot(kind='bar',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_habitaciones_metros_cubiertos.set_title("Metros Cubiertos Promedio según cantidad de Habitaciones",fontsize=20)

grafico_habitaciones_metros_cubiertos.set_xlabel("Habitaciones",fontsize=14)

grafico_habitaciones_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
antiguedad_metros_cubiertos=filtrado_metros[['antiguedad','metroscubiertos']].copy()

antiguedad_metros_cubiertos.dropna(inplace=True)

group_antiguedad_metros_cubiertos=antiguedad_metros_cubiertos.groupby(['antiguedad']).agg({'metroscubiertos':'mean'})

group_antiguedad_metros_cubiertos.head()
grafico_antiguedad_metros_cubiertos=group_antiguedad_metros_cubiertos.plot(kind='line',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_antiguedad_metros_cubiertos.set_title("Metros Cubiertos Promedio según la Antigüedad",fontsize=20)

grafico_antiguedad_metros_cubiertos.set_xlabel("Antigüedad",fontsize=14)

grafico_antiguedad_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
baños_metros_cubiertos=filtrado_metros[['banos','metroscubiertos']].copy()

baños_metros_cubiertos.dropna(inplace=True)

baños_metros_cubiertos['banos']=baños_metros_cubiertos['banos'].astype('int32')

group_baños_metros_cubiertos=baños_metros_cubiertos.groupby(['banos']).agg({'metroscubiertos':'mean'})

group_baños_metros_cubiertos
grafico_baños_metros_cubiertos=group_baños_metros_cubiertos.plot(kind='bar',color='tomato',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_baños_metros_cubiertos.set_title("Metros Cubiertos Promedio según cantidad de Baños",fontsize=20)

grafico_baños_metros_cubiertos.set_xlabel("Baños",fontsize=14)

grafico_baños_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
garage_metros_cubiertos=filtrado_metros[['garages','metroscubiertos']].copy()

garage_metros_cubiertos.dropna(inplace=True)

garage_metros_cubiertos['garages']=garage_metros_cubiertos['garages'].astype('int32')

group_garage_metros_cubiertos=garage_metros_cubiertos.groupby(['garages']).agg({'metroscubiertos':'mean'})

group_garage_metros_cubiertos
grafico_garage_metros_cubiertos=group_garage_metros_cubiertos.plot(kind='bar',color='purple',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_garage_metros_cubiertos.set_title("Metros Cubiertos Promedio según Cantidad de Garajes",fontsize=20)

grafico_garage_metros_cubiertos.set_xlabel("Garajes",fontsize=14)

grafico_garage_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
gimnasio_metros_cubiertos=filtrado_metros[['gimnasio','metroscubiertos']].copy()

gimnasio_metros_cubiertos.dropna(inplace=True)

gimnasio_metros_cubiertos['gimnasio'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_gimnasio_metros_cubiertos=gimnasio_metros_cubiertos.groupby(['gimnasio']).agg({'metroscubiertos':'mean'})

group_gimnasio_metros_cubiertos
grafico_gimnasio_metros_cubiertos=group_gimnasio_metros_cubiertos.plot(kind='bar',color='chocolate',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_gimnasio_metros_cubiertos.set_title("Metros Cubiertos Promedio según si hay Gimnasio",fontsize=20)

grafico_gimnasio_metros_cubiertos.set_xlabel("¿Tiene Gimansio?",fontsize=14)

grafico_gimnasio_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
piscina_metros_cubiertos=filtrado_metros[['piscina','metroscubiertos']].copy()

piscina_metros_cubiertos.dropna(inplace=True)

piscina_metros_cubiertos['piscina'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_piscina_metros_cubiertos=piscina_metros_cubiertos.groupby(['piscina']).agg({'metroscubiertos':'mean'})

group_piscina_metros_cubiertos
grafico_piscina_metros_cubiertos=group_piscina_metros_cubiertos.plot(kind='bar',color='darkturquoise',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_piscina_metros_cubiertos.set_title("Metros Cubiertos Promedio según si hay Piscina",fontsize=20)

grafico_piscina_metros_cubiertos.set_xlabel("¿Tiene Piscina?",fontsize=14)

grafico_piscina_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
sum_metros_cubiertos=filtrado_metros[['usosmultiples','metroscubiertos']].copy()

sum_metros_cubiertos.dropna(inplace=True)

sum_metros_cubiertos['usosmultiples'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_sum_metros_cubiertos=sum_metros_cubiertos.groupby(['usosmultiples']).agg({'metroscubiertos':'mean'})

group_sum_metros_cubiertos
grafico_sum_metros_cubiertos=group_sum_metros_cubiertos.plot(kind='bar',color='greenyellow',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_sum_metros_cubiertos.set_title("Metros Cubiertos Promedio según si hay SUM",fontsize=20)

grafico_sum_metros_cubiertos.set_xlabel("¿Tiene SUM?",fontsize=14)

grafico_sum_metros_cubiertos.set_ylabel("Metros Cubiertos Promedio",fontsize=14)
tipo_propiedad_cubiertos=tipo_propiedad_metros.groupby(['tipodepropiedad']).agg({'metroscubiertos':'mean'})

tipo_propiedad_cubiertos
plt.subplots(figsize=(10,10))

grafico_tipo_cubiertos=sns.barplot(x=tipo_propiedad_cubiertos['metroscubiertos'],y=tipo_propiedad_cubiertos.index,orient='h',palette='tab20')

grafico_tipo_cubiertos.set_title("Promedio de Metros Cubiertos por Top 10 de Tipo de Propiedad",fontsize=20)

grafico_tipo_cubiertos.set_xlabel("Promedio de Metros Cubiertos",fontsize=12)

grafico_tipo_cubiertos.set_ylabel("Tipo de Propiedad",fontsize=12)
pivot_gimnasio_tipo_metroscubiertos=gimnasio_tipo_metros.pivot_table(index='tipodepropiedad',columns='gimnasio',values='metroscubiertos',aggfunc='mean')

pivot_gimnasio_tipo_metroscubiertos.dropna(inplace=True)

pivot_gimnasio_tipo_metroscubiertos
grafico_gimnasio_tipo_metroscubiertos=pivot_gimnasio_tipo_metroscubiertos.plot(kind='bar',color=['chocolate','royalblue'],rot=70,fontsize=12,figsize=(10,10))

grafico_gimnasio_tipo_metroscubiertos.set_title("Promedio de Metros Cubiertos por Tipo de Propiedad según si tiene Gimnasio",fontsize=20)

grafico_gimnasio_tipo_metroscubiertos.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_gimnasio_tipo_metroscubiertos.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Gimnasio?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
pivot_piscina_tipo_metroscubiertos=piscina_tipo_metros.pivot_table(index='tipodepropiedad',columns='piscina',values='metroscubiertos',aggfunc='mean')

pivot_piscina_tipo_metroscubiertos.dropna(inplace=True)

pivot_piscina_tipo_metroscubiertos
grafico_piscina_tipo_metroscubiertos=pivot_piscina_tipo_metroscubiertos.plot(kind='bar',color=['tomato','darkturquoise'],rot=70,fontsize=12,figsize=(10,10))

grafico_piscina_tipo_metroscubiertos.set_title("Promedio de Metros Cubiertos por Tipo de Propiedad según si tiene Piscina",fontsize=20)

grafico_piscina_tipo_metroscubiertos.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_piscina_tipo_metroscubiertos.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Piscina?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
pivot_sum_tipo_metroscubiertos=sum_tipo_metros.pivot_table(index='tipodepropiedad',columns='usosmultiples',values='metroscubiertos',aggfunc='mean')

pivot_sum_tipo_metroscubiertos.dropna(inplace=True)

pivot_sum_tipo_metroscubiertos
grafico_sum_tipo_metroscubiertos=pivot_sum_tipo_metroscubiertos.plot(kind='bar',color=['saddlebrown','green'],rot=70,fontsize=12,figsize=(10,10))

grafico_sum_tipo_metroscubiertos.set_title("Promedio de Metros Cubiertos por Tipo de Propiedad según si tiene SUM",fontsize=20)

grafico_sum_tipo_metroscubiertos.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_sum_tipo_metroscubiertos.set_ylabel("Promedio de Metros",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene SUM?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
dif_metros=filtrado_metros.groupby(['provincia']).agg({'difmetros':'mean'})

dif_metros.reset_index(inplace=True)

dif_metros
dif_metros_mex=pd.merge(mex,dif_metros,on='provincia',how='left')

dif_metros_mex.fillna(0,inplace=True)

dif_metros_mex.head()
fig, grafico_dif_metros_provincia = plt.subplots(1, figsize=(20, 10))

dif_metros_mex.plot(column='difmetros',cmap='YlGnBu', edgecolor='black',ax=grafico_dif_metros_provincia,legend=True,figsize=(30,30),linewidth=0.5)

grafico_dif_metros_provincia.axis('off')

grafico_dif_metros_provincia.set_title("Promedio de Metros Descubiertos por Estado",fontsize=20)
habitaciones_metros_difmetros=filtrado_metros[['habitaciones','difmetros']].copy()

habitaciones_metros_difmetros.dropna(inplace=True)

habitaciones_metros_difmetros['habitaciones']=habitaciones_metros_difmetros['habitaciones'].astype('int32')

group_habitaciones_metros_difmetros=habitaciones_metros_difmetros.groupby(['habitaciones']).agg({'difmetros':'mean'})

group_habitaciones_metros_difmetros
grafico_habitaciones_metros_difmetros=group_habitaciones_metros_difmetros.plot(kind='bar',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_habitaciones_metros_difmetros.set_title("Promedio de Metros Descubiertos según cantidad de Habitaciones",fontsize=20)

grafico_habitaciones_metros_difmetros.set_xlabel("Habitaciones",fontsize=14)

grafico_habitaciones_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
antiguedad_metros_difmetros=filtrado_metros[['antiguedad','difmetros']].copy()

antiguedad_metros_difmetros.dropna(inplace=True)

group_antiguedad_metros_difmetros=antiguedad_metros_difmetros.groupby(['antiguedad']).agg({'difmetros':'mean'})

group_antiguedad_metros_difmetros.head()
grafico_antiguedad_metros_difmetros=group_antiguedad_metros_difmetros.plot(kind='line',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_antiguedad_metros_difmetros.set_title("Promedio de Metros Descubiertos según la Antigüedad",fontsize=20)

grafico_antiguedad_metros_difmetros.set_xlabel("Antigüedad",fontsize=14)

grafico_antiguedad_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
baños_metros_difmetros=filtrado_metros[['banos','difmetros']].copy()

baños_metros_difmetros.dropna(inplace=True)

baños_metros_difmetros['banos']=baños_metros_difmetros['banos'].astype('int32')

group_baños_metros_difmetros=baños_metros_difmetros.groupby(['banos']).agg({'difmetros':'mean'})

group_baños_metros_difmetros
grafico_baños_metros_difmetros=group_baños_metros_difmetros.plot(kind='bar',color='tomato',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_baños_metros_difmetros.set_title("Promedio de Metros Descubiertos según cantidad de Baños",fontsize=20)

grafico_baños_metros_difmetros.set_xlabel("Baños",fontsize=14)

grafico_baños_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
garage_metros_difmetros=filtrado_metros[['garages','difmetros']].copy()

garage_metros_difmetros.dropna(inplace=True)

garage_metros_difmetros['garages']=garage_metros_difmetros['garages'].astype('int32')

group_garage_metros_difmetros=garage_metros_difmetros.groupby(['garages']).agg({'difmetros':'mean'})

group_garage_metros_difmetros
grafico_garage_metros_difmetros=group_garage_metros_difmetros.plot(kind='bar',color='purple',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_garage_metros_difmetros.set_title("Promedio de Metros Descubiertos según cantidad de Garajes",fontsize=20)

grafico_garage_metros_difmetros.set_xlabel("Garajes",fontsize=14)

grafico_garage_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
gimnasio_metros_difmetros=filtrado_metros[['gimnasio','difmetros']].copy()

gimnasio_metros_difmetros.dropna(inplace=True)

gimnasio_metros_difmetros['gimnasio'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_gimnasio_metros_difmetros=gimnasio_metros_difmetros.groupby(['gimnasio']).agg({'difmetros':'mean'})

group_gimnasio_metros_difmetros
grafico_gimnasio_metros_difmetros=group_gimnasio_metros_difmetros.plot(kind='bar',color='chocolate',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_gimnasio_metros_difmetros.set_title("Promedio de Metros Descubiertos según si hay Gimnasio",fontsize=20)

grafico_gimnasio_metros_difmetros.set_xlabel("¿Tiene Gimansio?",fontsize=14)

grafico_gimnasio_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
piscina_metros_difmetros=filtrado_metros[['piscina','difmetros']].copy()

piscina_metros_difmetros.dropna(inplace=True)

piscina_metros_difmetros['piscina'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_piscina_metros_difmetros=piscina_metros_difmetros.groupby(['piscina']).agg({'difmetros':'mean'})

group_piscina_metros_difmetros
grafico_piscina_metros_difmetros=group_piscina_metros_difmetros.plot(kind='bar',color='darkturquoise',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_piscina_metros_difmetros.set_title("Promedio de Metros Descubiertos según si hay Piscina",fontsize=20)

grafico_piscina_metros_difmetros.set_xlabel("¿Tiene Piscina?",fontsize=14)

grafico_piscina_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
sum_metros_difmetros=filtrado_metros[['usosmultiples','difmetros']].copy()

sum_metros_difmetros.dropna(inplace=True)

sum_metros_difmetros['usosmultiples'].replace({0.0:'NO',1.0:'SI'},inplace=True)

group_sum_metros_difmetros=sum_metros_difmetros.groupby(['usosmultiples']).agg({'difmetros':'mean'})

group_sum_metros_difmetros
grafico_sum_metros_difmetros=group_sum_metros_difmetros.plot(kind='bar',color='greenyellow',fontsize=12,figsize=(10,10),rot=0,legend=False)

grafico_sum_metros_difmetros.set_title("Promedio de Metros Descubiertos según si hay SUM",fontsize=20)

grafico_sum_metros_difmetros.set_xlabel("¿Tiene SUM?",fontsize=14)

grafico_sum_metros_difmetros.set_ylabel("Promedio de Metros Descubiertos",fontsize=14)
tipo_propiedad_descubiertos=tipo_propiedad_metros.groupby(['tipodepropiedad']).agg({'difmetros':'mean'})

tipo_propiedad_descubiertos
plt.subplots(figsize=(10,10))

grafico_tipo_descubiertos=sns.barplot(x=tipo_propiedad_descubiertos['difmetros'],y=tipo_propiedad_descubiertos.index,orient='h',palette='tab20b')

grafico_tipo_descubiertos.set_title("Promedio de Metros Descubiertos por Top 10 de Tipo de Propiedad",fontsize=20)

grafico_tipo_descubiertos.set_xlabel("Promedio de Metros Descubiertos",fontsize=12)

grafico_tipo_descubiertos.set_ylabel("Tipo de Propiedad",fontsize=12)
preciom2=propiedades.loc[:,['ciudad','provincia','metrostotales','metroscubiertos','precio','Anio','Mes','Dia']].copy()

preciom2['precio m2 construccion']=(preciom2['precio']/preciom2['metroscubiertos'])

preciom2['preciom2']=(preciom2['precio']/preciom2['metrostotales'])

preciom2.head()
#tomando la cotizacion dolar hoy a 1USD = 19.40 pesos mexicanos

cotizacion = 19.4

preciom2['precio m2 dolares'] = (preciom2['preciom2']/cotizacion)

preciom2['precio m2 construccion dolares'] = (preciom2['precio m2 construccion']/cotizacion)

preciom2.head()
preciom2.size
preciom2.dropna(inplace = True)

preciom2.size
#cuidades/provincias con valor m2 mas caro

precio_ciudades = preciom2.groupby(['ciudad']).agg({'precio m2 dolares':[np.mean, np.size]})

precio_ciudades=precio_ciudades[precio_ciudades[('precio m2 dolares', 'size')] >= 100]

top10_m2_ciudad=precio_ciudades[[('precio m2 dolares', 'mean')]].sort_values(('precio m2 dolares', 'mean'), ascending = False)[:10]
f = plt.figure()



plt.title('Las Diez Cuidades con Precio del Metro Cuadrado mas Alto', color = 'black',fontsize=25)



ax = top10_m2_ciudad.reset_index().plot(kind='bar', x = 'ciudad', stacked=True, figsize=(16,8) ,alpha=0.75,\

                                 ax=f.gca(),rot=70)

handles, labels = ax.get_legend_handles_labels()



plt.legend(handles[::-1], labels[::-1], loc='upper right')



ax.set_ylabel('Precio promedio metro cuadrado',fontsize=20)

ax.set_xlabel('Ciudad',fontsize=20)

plt.show()
precio_provincias=preciom2.groupby(['provincia']).agg({'precio m2 dolares':[np.mean, np.size]})

precio_provincias=precio_provincias[precio_provincias[('precio m2 dolares', 'size')] >= 100]

top10_m2_provincia=precio_provincias[[('precio m2 dolares', 'mean')]].sort_values(('precio m2 dolares', 'mean'), ascending = False)[:10]
f = plt.figure()



plt.title('Los Diez Estados con Precio del Metro Cuadrado más Alto', color = 'black',fontsize=25)



ax = top10_m2_provincia.reset_index().plot(kind='bar', x = 'provincia', stacked=True, figsize=(16,8) ,alpha=0.75,\

                                 ax=f.gca(),rot=60,fontsize=20)

handles, labels = ax.get_legend_handles_labels()



plt.legend(handles[::-1], labels[::-1], loc='upper right')



ax.set_ylabel('Precio Promedio Metro Cuadrado',fontsize=20)

ax.set_xlabel('Estado',fontsize=20)

plt.show()

preciom2_anio=preciom2.groupby('Anio').agg({'precio m2 construccion dolares':[np.mean,np.size],'precio m2 dolares':[np.mean,np.size]})

preciom2_anio
#m2

l = [2012, 2013, 2014, 2015, 2016]

ax = preciom2_anio.plot(kind='line', y = ('precio m2 dolares', 'mean'), figsize=(15,8),\

             xticks=l, legend = False, color = 'indianred', lw = 4,fontsize=15)

ax.set_ylabel('$ USD',fontsize=15)

ax.set_xlabel('Año',fontsize=15)

ax.set_ylim([0,1000])

ax.set_title("Variacion del Precio del Metro Cuadrado (USD)",fontsize=20)



plt.show()
#variacion anio por anio a ver si se ve algo interesante....

preciom2_2012=preciom2[preciom2['Anio']==2012]

preciom2_2013=preciom2[preciom2['Anio']==2013]

preciom2_2014=preciom2[preciom2['Anio']==2014]

preciom2_2015=preciom2[preciom2['Anio']==2015]

preciom2_2016=preciom2[preciom2['Anio']==2016]
preciom2_2012 = preciom2_2012.groupby('Mes').agg({'precio m2 dolares':[np.mean, np.size]})

preciom2_2013 = preciom2_2013.groupby('Mes').agg({'precio m2 dolares':[np.mean, np.size]})

preciom2_2014 = preciom2_2014.groupby('Mes').agg({'precio m2 dolares':[np.mean, np.size]})

preciom2_2015 = preciom2_2015.groupby('Mes').agg({'precio m2 dolares':[np.mean, np.size]})

preciom2_2016 = preciom2_2016.groupby('Mes').agg({'precio m2 dolares':[np.mean, np.size]})
fig = plt.figure(figsize=(12, 6));



ax = fig.add_axes([0,0,1,1]);



preciom2_2016['precio m2 dolares', 'mean'].plot.line(c='red', label="2016", xticks = [i for i in range(1, 13)], lw = 4,fontsize=15);

preciom2_2015['precio m2 dolares', 'mean'].plot.line(c='gold', label="2015", lw = 4);

preciom2_2014['precio m2 dolares', 'mean'].plot.line(c='chartreuse', label="2014", lw = 4);

preciom2_2013['precio m2 dolares', 'mean'].plot.line(c='mediumspringgreen', label="2013", lw = 4);

preciom2_2012['precio m2 dolares', 'mean'].plot.line(c='navy', label="2012", lw = 4);



plt.title("Comprativa de la Variacion del Precio por Metro Cuadrado por Mes entre 2012 y 2016", fontsize=20)

ax.set_ylim([0,1500])

ax.set_xlim([1, 12])

ax.set_ylabel('$ USD',fontsize=20)

ax.set_xlabel('Mes',fontsize=20)





leyenda=plt.legend(fontsize=14,frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.set_title('Año',prop=dict(size=14))

leyenda.get_frame().set_linewidth(1.0)
direccion=propiedades[['provincia','tipodepropiedad','direccion','precio']].copy()

direccion.dropna(inplace=True)

direccion['avenida']=direccion.direccion.str.contains('AV|av|Avenida|avenida')

direccion.head()
cantidad_direcciones_avenida=direccion['avenida'].value_counts().reset_index()

cantidad_direcciones_avenida.replace({False:'NO',True:'SI'},inplace=True)

cantidad_direcciones_avenida.set_index('index',inplace=True)

cantidad_direcciones_avenida
plt.subplots(figsize=(10,10))

grafico_avenida=sns.barplot(x=cantidad_direcciones_avenida['avenida'],y=cantidad_direcciones_avenida.index,orient='h',palette='magma')

grafico_avenida.set_title("Cantidad de Propiedades en Avenidas",fontsize=20)

grafico_avenida.set_xlabel("Cantidad",fontsize=12)

grafico_avenida.set_ylabel("¿Está en una Avenida?",fontsize=12)
propiedades_avenida=direccion[(direccion['avenida']==True)]

tipo_avenida=propiedades_avenida['tipodepropiedad'].value_counts().head(8)

tipo_avenida
plt.subplots(figsize=(10,10))

grafico_tipo_avenida=sns.barplot(x=tipo_avenida.values,y=tipo_avenida.index,orient='h',palette='gnuplot')

grafico_tipo_avenida.set_title("Tipo de Propiedades en Avenida",fontsize=20)

grafico_tipo_avenida.set_xlabel("Cantidad",fontsize=12)

grafico_tipo_avenida.set_ylabel("Tipo de Propiedad",fontsize=12)
cantidad_provincia_avenida=propiedades_avenida['provincia'].value_counts().reset_index()

cantidad_provincia_avenida.rename(columns={'index':'provincia','provincia':'cantidad'},inplace=True)

cantidad_provincia_avenida
provincia_avenida=pd.merge(mex,cantidad_provincia_avenida,on='provincia',how='left')

provincia_avenida.fillna(0,inplace=True)

provincia_avenida.head()
fig, grafico_provincia_avenida = plt.subplots(1, figsize=(20, 10))

provincia_avenida.plot(column='cantidad',cmap='RdGy', edgecolor='black',ax=grafico_provincia_avenida,legend=True,figsize=(30,30),linewidth=0.5)

grafico_provincia_avenida.axis('off')

grafico_provincia_avenida.set_title("Cantidad de Propiedades sobre Avenidas por Estado",fontsize=22)
precio_avenida=direccion.groupby(['avenida']).agg({'precio':'mean'})

precio_avenida.reset_index(inplace=True)

precio_avenida['avenida'].replace({False:'NO',True:'SI'},inplace=True)

precio_avenida.set_index('avenida',inplace=True)

precio_avenida
plt.subplots(figsize=(10,10))

grafico_precio_avenida=sns.barplot(x=precio_avenida['precio'],y=precio_avenida.index,orient='h',palette='gnuplot')

grafico_precio_avenida.set_title("Precio Promedio por Propiedades en Avenida",fontsize=20)

grafico_precio_avenida.set_xlabel("Precio Promedio",fontsize=12)

grafico_precio_avenida.set_ylabel("¿Está en Avenida?",fontsize=12)
baños_hab_garages_tipo=propiedades[['tipodepropiedad','habitaciones','garages','banos']].copy()

baños_hab_garages_tipo.dropna(inplace=True)

baños_hab_garages_tipo['Valor']=1

baños_hab_garages_tipo['Valor']=baños_hab_garages_tipo['Valor'].astype('int32')

baños_hab_garages_tipo['habitaciones']=baños_hab_garages_tipo['habitaciones'].astype('int32')

baños_hab_garages_tipo['garages']=baños_hab_garages_tipo['garages'].astype('int32')

baños_hab_garages_tipo['banos']=baños_hab_garages_tipo['banos'].astype('int32')

baños_tipo=baños_hab_garages_tipo.pivot_table(index='tipodepropiedad',columns='banos',values='Valor',aggfunc='sum')

habitaciones_tipo=baños_hab_garages_tipo.pivot_table(index='tipodepropiedad',columns='habitaciones',values='Valor',aggfunc='sum')

garages_tipo=baños_hab_garages_tipo.pivot_table(index='tipodepropiedad',columns='garages',values='Valor',aggfunc='sum')

baños_tipo.fillna(0,inplace=True)

habitaciones_tipo.fillna(0,inplace=True)

garages_tipo.fillna(0,inplace=True)
plt.subplots(figsize=(12,12))

grafico_baño_tipo=sns.heatmap(baños_tipo,linewidths=.5,fmt=".0f",annot=True,cmap="CMRmap")

grafico_baño_tipo.set_title("Cantidad de Baños por Tipo de Propiedad",fontsize=20)

grafico_baño_tipo.set_xlabel("Cantidad de Baños",fontsize=12)

grafico_baño_tipo.set_ylabel("Tipos de Propiedades",fontsize=12)

grafico_baño_tipo.set_yticklabels(grafico_baño_tipo.get_yticklabels(),rotation=0)
plt.subplots(figsize=(12,12))

grafico_habitaciones_tipo=sns.heatmap(habitaciones_tipo,linewidths=.5,fmt=".0f",annot=True,cmap="tab20c_r")

grafico_habitaciones_tipo.set_title("Cantidad de Habitaciones por Tipo de Propiedad",fontsize=20)

grafico_habitaciones_tipo.set_xlabel("Cantidad de Habitaciones",fontsize=12)

grafico_habitaciones_tipo.set_ylabel("Tipos de Propiedades",fontsize=12)

grafico_habitaciones_tipo.set_yticklabels(grafico_habitaciones_tipo.get_yticklabels(),rotation=0)
plt.subplots(figsize=(12,12))

grafico_garages_tipo=sns.heatmap(garages_tipo,linewidths=.5,fmt=".0f",annot=True,cmap="inferno_r")

grafico_garages_tipo.set_title("Cantidad de Garajes por Tipo de Propiedad",fontsize=20)

grafico_garages_tipo.set_xlabel("Cantidad de Garajes",fontsize=12)

grafico_garages_tipo.set_ylabel("Tipos de Propiedades",fontsize=12)

grafico_garages_tipo.set_yticklabels(grafico_garages_tipo.get_yticklabels(),rotation=0)
piscina_tipo=propiedades[['tipodepropiedad','piscina']].copy()

piscina_tipo_top=pd.merge(piscina_tipo,tipo_propiedad_top,on='tipodepropiedad',how='inner')

piscina_tipo_top['piscina'].replace({0:'NO',1:'SI'},inplace=True)

piscina_tipo_top['Valor']=1

piscina_tipo_top.dropna(inplace=True)

group_piscina_top=piscina_tipo_top.pivot_table(index='tipodepropiedad',columns='piscina',values='Valor',aggfunc='sum')

group_piscina_top.fillna(0,inplace=True)

group_piscina_top
grafico_piscina_tipo=group_piscina_top.plot(kind='bar',rot=70,fontsize=12,figsize=(10,10))

grafico_piscina_tipo.set_title("Cantidad de Propiedades con Piscina según Tipo (Top 10)",fontsize=20)

grafico_piscina_tipo.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_piscina_tipo.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Piscina?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
gimnasio_tipo=propiedades[['tipodepropiedad','gimnasio']].copy()

gimnasio_tipo_top=pd.merge(gimnasio_tipo,tipo_propiedad_top,on='tipodepropiedad',how='inner')

gimnasio_tipo_top['gimnasio'].replace({0:'NO',1:'SI'},inplace=True)

gimnasio_tipo_top['Valor']=1

gimnasio_tipo_top.dropna(inplace=True)

group_gimnasio_top=gimnasio_tipo_top.pivot_table(index='tipodepropiedad',columns='gimnasio',values='Valor',aggfunc='sum')

group_gimnasio_top.fillna(0,inplace=True)

group_gimnasio_top
grafico_gimnasio_tipo=group_gimnasio_top.plot(kind='bar',color=['chocolate','orchid'],rot=70,fontsize=12,figsize=(10,10))

grafico_gimnasio_tipo.set_title("Cantidad de Propiedades con Gimnasio según Tipo (Top 10)",fontsize=20)

grafico_gimnasio_tipo.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_gimnasio_tipo.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene Gimnasio?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
sum_tipo=propiedades[['tipodepropiedad','usosmultiples']].copy()

sum_tipo_top=pd.merge(sum_tipo,tipo_propiedad_top,on='tipodepropiedad',how='inner')

sum_tipo_top['usosmultiples'].replace({0:'NO',1:'SI'},inplace=True)

sum_tipo_top['Valor']=1

sum_tipo_top.dropna(inplace=True)

group_sum_top=sum_tipo_top.pivot_table(index='tipodepropiedad',columns='usosmultiples',values='Valor',aggfunc='sum')

group_sum_top.fillna(0,inplace=True)

group_sum_top
grafico_sum_tipo=group_sum_top.plot(kind='bar',color=['tomato','darkturquoise'],rot=70,fontsize=12,figsize=(10,10))

grafico_sum_tipo.set_title("Cantidad de Propiedades con SUM según Tipo (Top 10)",fontsize=20)

grafico_sum_tipo.set_xlabel("Tipo de Propiedad",fontsize=14)

grafico_sum_tipo.set_ylabel("Cantidad",fontsize=14)

leyenda=plt.legend(['NO','SI'],fontsize=10,title='¿Tiene SUM?',frameon=True,facecolor='white',edgecolor='black',loc='best')

leyenda.get_frame().set_linewidth(1.0)
tipo_antiguedad=propiedades[['tipodepropiedad','antiguedad']].copy()

tipo_antiguedad_top=pd.merge(tipo_antiguedad,tipo_propiedad_top,on='tipodepropiedad',how='inner')

tipo_antiguedad_top.dropna(inplace=True)

tipo_antiguedad_top['antiguedad'].fillna(0,inplace=True)

group_tipo_antiguedad_top=tipo_antiguedad_top.groupby(['tipodepropiedad']).agg({'antiguedad':'mean'})

group_tipo_antiguedad_top
plt.subplots(figsize=(10,10))

grafico_tipo_antiguedad_top=sns.barplot(x=group_tipo_antiguedad_top['antiguedad'],y=group_tipo_antiguedad_top.index,orient='h',palette='magma')

grafico_tipo_antiguedad_top.set_title("Antigüedad Promedio del Top 10 de Tipo de Propiedades",fontsize=20)

grafico_tipo_antiguedad_top.set_xlabel("Antigüedad",fontsize=12)

grafico_tipo_antiguedad_top.set_ylabel("Tipo de Propiedad",fontsize=12)
tipo_habitaciones=propiedades[['tipodepropiedad','habitaciones']].copy()

tipo_habitaciones_top=pd.merge(tipo_habitaciones,tipo_propiedad_top,on='tipodepropiedad',how='inner')

tipo_habitaciones_top.dropna(inplace=True)

tipo_habitaciones_top['habitaciones'].fillna(0,inplace=True)

group_tipo_habitaciones_top=tipo_habitaciones_top.groupby(['tipodepropiedad']).agg({'habitaciones':'median'})

group_tipo_habitaciones_top
plt.subplots(figsize=(10,10))

grafico_tipo_habitaciones_top=sns.barplot(x=group_tipo_habitaciones_top['habitaciones'],y=group_tipo_habitaciones_top.index,orient='h',palette='magma')

grafico_tipo_habitaciones_top.set_title('Mediana de Habitaciones del Top 10 de Tipo de Propiedades', fontsize=24);

grafico_tipo_habitaciones_top.set_xlabel("Habitaciones",fontsize=12)

grafico_tipo_habitaciones_top.set_ylabel("Tipo de Propiedad",fontsize=12)
tipo_banos=propiedades[['tipodepropiedad','banos']].copy()

tipo_banos_top=pd.merge(tipo_banos,tipo_propiedad_top,on='tipodepropiedad',how='inner')

tipo_banos_top.dropna(inplace=True)

tipo_banos_top['banos'].fillna(0,inplace=True)

group_tipo_banos_top=tipo_banos_top.groupby(['tipodepropiedad']).agg({'banos':'median'})

group_tipo_banos_top
plt.subplots(figsize=(10,10))

grafico_tipo_banos_top=sns.barplot(x=group_tipo_banos_top['banos'],y=group_tipo_banos_top.index,orient='h',palette='magma')

grafico_tipo_banos_top.set_title('Mediana de Baños del Top 10 de Tipo de Propiedades', fontsize=24);

grafico_tipo_banos_top.set_xlabel("Baños",fontsize=12)

grafico_tipo_banos_top.set_ylabel("Tipo de Propiedad",fontsize=12)

plt.xticks([0,1,2,3])
cantidad_banos=propiedades['banos'].value_counts().sort_index().reset_index()

cantidad_banos['index']=cantidad_banos['index'].astype('int32')

cantidad_banos['banos']=cantidad_banos['banos'].astype('int32')

cantidad_banos.set_index('index',inplace=True)

cantidad_banos
plt.subplots(figsize=(10,10))

grafico_cantidad_banos=sns.barplot(x=cantidad_banos['banos'],y=cantidad_banos.index,orient='h',palette='Dark2')

grafico_cantidad_banos.set_title("Cantidad de Propiedades por Cantidad de Baños",fontsize=20)

grafico_cantidad_banos.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_banos.set_ylabel("Cantidad de Baños",fontsize=15)
cantidad_habitaciones=propiedades['habitaciones'].value_counts().sort_index().reset_index()

cantidad_habitaciones['index']=cantidad_habitaciones['index'].astype('int32')

cantidad_habitaciones.set_index('index',inplace=True)

cantidad_habitaciones_top=cantidad_habitaciones.head(5)

cantidad_habitaciones_resto=cantidad_habitaciones.tail(5)

cantidad_habitaciones_top
cantidad_habitaciones_resto
grafico_cantidad_habitaciones_top=cantidad_habitaciones_top.plot(kind='bar',rot=0,legend=False,fontsize=12,figsize=(10,10))

grafico_cantidad_habitaciones_top.set_title("Cantidad de Propiedades por Cantidad de Habitaciones (Top)",fontsize=20)

grafico_cantidad_habitaciones_top.set_xlabel("Cantidad de Habitaciones",fontsize=12)

grafico_cantidad_habitaciones_top.set_ylabel("Cantidad de Propiedades",fontsize=12)
grafico_cantidad_habitaciones_resto=cantidad_habitaciones_resto.plot(kind='bar',color='tomato',rot=0,legend=False,fontsize=12,figsize=(10,10))

grafico_cantidad_habitaciones_resto.set_title("Cantidad de Propiedades por Cantidad de Habitaciones (Resto)",fontsize=20)

grafico_cantidad_habitaciones_resto.set_xlabel("Cantidad de Habitaciones",fontsize=12)

grafico_cantidad_habitaciones_resto.set_ylabel("Cantidad de Propiedades",fontsize=12)
cantidad_garages=propiedades['garages'].value_counts().sort_index().reset_index()

cantidad_garages['index']=cantidad_garages['index'].astype('int32')

cantidad_garages['garages']=cantidad_garages['garages'].astype('int32')

cantidad_garages.set_index('index',inplace=True)

cantidad_garages
plt.subplots(figsize=(10,10))

grafico_cantidad_garages=sns.barplot(x=cantidad_garages['garages'],y=cantidad_garages.index,orient='h',palette='rainbow')

grafico_cantidad_garages.set_title("Cantidad de Propiedades segun Cantidad de Garajes",fontsize=20)

grafico_cantidad_garages.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_garages.set_ylabel("Cantidad de Garajes",fontsize=15)
cantidad_piscina=propiedades['piscina'].value_counts().reset_index()

cantidad_piscina['index'].replace({0:'NO',1:'SI'},inplace=True)

cantidad_piscina.set_index('index',inplace=True)

cantidad_piscina
plt.subplots(figsize=(10,10))

grafico_cantidad_piscina=sns.barplot(x=cantidad_piscina['piscina'],y=cantidad_piscina.index,orient='h',palette='cool')

grafico_cantidad_piscina.set_title("Cantidad de Propiedades con Piscina",fontsize=20)

grafico_cantidad_piscina.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_piscina.set_ylabel("¿Tiene Piscina?",fontsize=15)
cantidad_gimnasio=propiedades['gimnasio'].value_counts().reset_index()

cantidad_gimnasio['index'].replace({0:'NO',1:'SI'},inplace=True)

cantidad_gimnasio.set_index('index',inplace=True)

cantidad_gimnasio
plt.subplots(figsize=(10,10))

grafico_cantidad_gimnasio=sns.barplot(x=cantidad_gimnasio['gimnasio'],y=cantidad_gimnasio.index,orient='h',palette='hot')

grafico_cantidad_gimnasio.set_title("Cantidad de Propiedades con Gimnasio",fontsize=20)

grafico_cantidad_gimnasio.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_gimnasio.set_ylabel("¿Tiene Gimnasio?",fontsize=15)
cantidad_sum=propiedades['usosmultiples'].value_counts().reset_index()

cantidad_sum['index'].replace({0:'NO',1:'SI'},inplace=True)

cantidad_sum.set_index('index',inplace=True)

cantidad_sum
plt.subplots(figsize=(10,10))

grafico_cantidad_sum=sns.barplot(x=cantidad_sum['usosmultiples'],y=cantidad_sum.index,orient='h',palette='winter')

grafico_cantidad_sum.set_title("Cantidad de Propiedades con SUM",fontsize=20)

grafico_cantidad_sum.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_sum.set_ylabel("¿Tiene SUM?",fontsize=15)
cantidad_escuelas=propiedades['escuelascercanas'].value_counts().reset_index()

cantidad_escuelas['index'].replace({0:'NO',1:'SI'},inplace=True)

cantidad_escuelas.set_index('index',inplace=True)

cantidad_escuelas
plt.subplots(figsize=(10,10))

grafico_cantidad_escuelas=sns.barplot(x=cantidad_escuelas['escuelascercanas'],y=cantidad_escuelas.index,orient='h',palette='brg')

grafico_cantidad_escuelas.set_title("Cantidad de Propiedades con Escuelas Cercanas",fontsize=20)

grafico_cantidad_escuelas.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_escuelas.set_ylabel("¿Tiene Escuelas Cercanas?",fontsize=15)
cantidad_centros=propiedades['centroscomercialescercanos'].value_counts().reset_index()

cantidad_centros['index'].replace({0:'NO',1:'SI'},inplace=True)

cantidad_centros.set_index('index',inplace=True)

cantidad_centros
plt.subplots(figsize=(10,10))

grafico_cantidad_centros=sns.barplot(x=cantidad_centros['centroscomercialescercanos'],y=cantidad_centros.index,orient='h',palette='CMRmap')

grafico_cantidad_centros.set_title("Cantidad de Propiedades con Centros Comerciales Cercanos",fontsize=20)

grafico_cantidad_centros.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_centros.set_ylabel("¿Tiene Centros Comeciales Cercanos?",fontsize=15)
cantidad_tipo_propiedad_top=propiedades['tipodepropiedad'].value_counts().head(10)

cantidad_tipo_propiedad_top
plt.subplots(figsize=(10,10))

grafico_cantidad_tipo_top=sns.barplot(x=cantidad_tipo_propiedad_top.values,y=cantidad_tipo_propiedad_top.index,orient='h',palette='CMRmap')

grafico_cantidad_tipo_top.set_title("Top 10 de Tipo de Propiedades",fontsize=20)

grafico_cantidad_tipo_top.set_xlabel("Cantidad de Propiedades",fontsize=15)

grafico_cantidad_tipo_top.set_ylabel("Tipo de Propiedad",fontsize=15)
cantidad_antiguedad=propiedades['antiguedad'].value_counts().sort_index().reset_index()

cantidad_antiguedad.drop(columns={'index'},inplace=True)

cantidad_antiguedad_top=cantidad_antiguedad.head(40)

cantidad_antiguedad_resto=cantidad_antiguedad.tail(38)
grafico_antiguedad_top=cantidad_antiguedad_top.plot(kind='bar',color='chocolate',fontsize=12,figsize=(15,15),rot=0,legend=False)

grafico_antiguedad_top.set_title("Cantidad de Propiedades según Antigüedad (Menor de 40 Años)",fontsize=30)

grafico_antiguedad_top.set_xlabel("Antigüedad",fontsize=20)

grafico_antiguedad_top.set_ylabel("Cantidad de Propiedades",fontsize=20)
grafico_antiguedad_resto=cantidad_antiguedad_resto.plot(kind='bar',color='r',fontsize=12,figsize=(15,15),rot=0,legend=False)

grafico_antiguedad_resto.set_title("Cantidad de Propiedades según Antigüedad (mayor de 40 Años)",fontsize=30)

grafico_antiguedad_resto.set_xlabel("Antigüedad",fontsize=20)

grafico_antiguedad_resto.set_ylabel("Cantidad de Propiedades",fontsize=20)
precio_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'precio':'mean'})

precio_ciudad=precio_ciudad.reset_index()

precio_ciudad=precio_ciudad.sort_values(by='precio', ascending=False)

precio_ciudad["ciudad_provincia"] = precio_ciudad["ciudad"].map(str) + ", " +precio_ciudad["provincia"]

precio_ciudad=precio_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=precio_ciudad['ciudad_provincia'],x=precio_ciudad['precio'],orient='h',palette='Wistia')

graf_precio_ciudad.set_title("Las 10 Ciudades más Caras",fontsize=20)

graf_precio_ciudad.set_xlabel("Precio",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)
piscina_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'piscina':'sum'})

piscina_ciudad=piscina_ciudad.reset_index()

piscina_ciudad=piscina_ciudad.sort_values(by='piscina', ascending=False)

piscina_ciudad["ciudad_provincia"] =piscina_ciudad["ciudad"].map(str) + ", " +piscina_ciudad["provincia"]

piscina_ciudad=piscina_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=piscina_ciudad['ciudad_provincia'],x=piscina_ciudad['piscina'],orient='h')

graf_precio_ciudad.set_title("Las 10 Ciudades con más Propiedades con Piscina",fontsize=20)

graf_precio_ciudad.set_xlabel("Cantidad de Propiedades",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)
cant_prop_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'tipodepropiedad':'count'})

cant_prop_ciudad=cant_prop_ciudad.reset_index()

cant_prop_ciudad=cant_prop_ciudad.sort_values(by='tipodepropiedad', ascending=False)

cant_prop_ciudad["ciudad_provincia"] =cant_prop_ciudad["ciudad"].map(str) + ", " +piscina_ciudad["provincia"]
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=cant_prop_ciudad['ciudad_provincia'],x=cant_prop_ciudad['tipodepropiedad'],orient='h',palette='rocket')

graf_precio_ciudad.set_title("Las 10 Ciudades con mayor Cantidad de Propiedades en Venta",fontsize=20)

graf_precio_ciudad.set_xlabel("Cantidad de Propiedades",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)
#Función para mostrar los valores en el barplot

#Fuente: https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values

def show_values_on_bars(axs, h_v="v", space=0.4):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center") 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height()

                value = int(p.get_width())

                ax.text(_x, _y, value, ha="left")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
metros_totales_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'metrostotales':'mean'})

metros_totales_ciudad=metros_totales_ciudad.reset_index()

metros_totales_ciudad=metros_totales_ciudad.sort_values(by='metrostotales', ascending=False)

metros_totales_ciudad["ciudad_provincia"] =metros_totales_ciudad["ciudad"].map(str) + ", " +metros_totales_ciudad["provincia"]

metros_totales_ciudad=metros_totales_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=metros_totales_ciudad['ciudad_provincia'],x=metros_totales_ciudad['metrostotales'],orient='h',palette='Accent')

graf_precio_ciudad.set_title("Las 10 Ciudades con Mayor Promedio de Metros Totales",fontsize=20)

graf_precio_ciudad.set_xlabel("Promedio de Metros Totales",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)

show_values_on_bars(graf_precio_ciudad, "h", 0.3)
metros_cubiertos_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'metroscubiertos':'mean'})

metros_cubiertos_ciudad=metros_cubiertos_ciudad.reset_index()

metros_cubiertos_ciudad=metros_cubiertos_ciudad.sort_values(by='metroscubiertos', ascending=False)

metros_cubiertos_ciudad["ciudad_provincia"] =metros_cubiertos_ciudad["ciudad"].map(str) + ", " +metros_cubiertos_ciudad["provincia"]

metros_cubiertos_ciudad=metros_cubiertos_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=metros_cubiertos_ciudad['ciudad_provincia'],x=metros_cubiertos_ciudad['metroscubiertos'],orient='h',palette='copper')

graf_precio_ciudad.set_title("Las 10 Ciudades con Mayor Promedio de Metros Cubiertos",fontsize=20)

graf_precio_ciudad.set_xlabel("Promedio de Metros Cubiertos",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)

show_values_on_bars(graf_precio_ciudad, "h", 0.3)
def metrosCubiertos(metros):

    if (metros>0):

        return metros

    else:

        return 0
propiedades['difmetros']=propiedades['metrostotales']-propiedades['metroscubiertos']

propiedades["metrosdescubiertos"] = propiedades.apply(lambda x: metrosCubiertos(x['difmetros']),axis=1)

propiedades.drop(columns={'difmetros'},inplace=True)

metros_descubiertos_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'metrosdescubiertos':'mean'})

metros_descubiertos_ciudad=metros_descubiertos_ciudad.reset_index()

metros_descubiertos_ciudad=metros_descubiertos_ciudad.sort_values(by='metrosdescubiertos', ascending=False)

metros_descubiertos_ciudad["ciudad_provincia"] =metros_descubiertos_ciudad["ciudad"].map(str) + ", " +metros_descubiertos_ciudad["provincia"]

metros_descubiertos_ciudad=metros_descubiertos_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=metros_descubiertos_ciudad['ciudad_provincia'],x=metros_descubiertos_ciudad['metrosdescubiertos'],orient='h',palette='cividis')

graf_precio_ciudad.set_title("Las 10 Ciudades con Mayor Promedio de Metros Descubiertos",fontsize=20)

graf_precio_ciudad.set_xlabel("Promedio de Metros Descubiertos",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)

show_values_on_bars(graf_precio_ciudad, "h", 0.3)
antiguedad_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'antiguedad':'mean'})

antiguedad_ciudad=antiguedad_ciudad.reset_index()

antiguedad_ciudad=antiguedad_ciudad.sort_values(by='antiguedad', ascending=False)

antiguedad_ciudad["ciudad_provincia"] =antiguedad_ciudad["ciudad"].map(str) + ", " +antiguedad_ciudad["provincia"]

antiguedad_ciudad=antiguedad_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=antiguedad_ciudad['ciudad_provincia'],x=antiguedad_ciudad['antiguedad'],orient='h',palette='tab10')

graf_precio_ciudad.set_title("Las 10 Ciudades con Mayor Promedio de Años de Antiguedad",fontsize=20)

graf_precio_ciudad.set_xlabel("Antiegüedad",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)
habitaciones_ciudad=propiedades.groupby(['ciudad','provincia']).agg({'habitaciones':'mean'})

habitaciones_ciudad=habitaciones_ciudad.reset_index()

habitaciones_ciudad=habitaciones_ciudad.sort_values(by='habitaciones', ascending=False)

habitaciones_ciudad["ciudad_provincia"] =habitaciones_ciudad["ciudad"].map(str) + ", " +habitaciones_ciudad["provincia"]

habitaciones_ciudad=habitaciones_ciudad.head(10)
plt.subplots(figsize=(10,10))

graf_precio_ciudad=sns.barplot(y=habitaciones_ciudad['ciudad_provincia'],x=habitaciones_ciudad['habitaciones'],orient='h')

graf_precio_ciudad.set_title("Las 10 Ciudades con Mayor Promedio de Habitaciones",fontsize=20)

graf_precio_ciudad.set_xlabel("Cantidad de Habitaciones",fontsize=15)

graf_precio_ciudad.set_ylabel("Ciudad",fontsize=15)