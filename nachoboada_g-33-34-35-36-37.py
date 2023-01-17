import seaborn as sns
# Start with loading all necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline

df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)

df['fecha'] = df['fecha'].astype('datetime64[ns]')

df.head(2).T
df.info()
df.sort_values(by='fecha', ascending=True, inplace=True)

df.head(50)
df.fecha.head(50)
df['anio']=df.fecha.dt.year
df['mes']=df.fecha.dt.month
df.info()
precioxmes=df[['anio','mes','precio']].groupby(['anio','mes']).agg(['mean','size']).reset_index()
a=precioxmes[['anio','mes']].astype(str)

a['anio']+'-'+a['mes']

precioxmes['aniomes']=a['anio']+'-'+a['mes']

precioxmes.drop('anio',axis=1)

precioxmes=precioxmes.drop(['mes','anio'],axis=1)



precioxmes.columns=['precio','cantidad','aniomes']

precioxmes
ax=precioxmes.plot(kind='bar',x='aniomes',y='precio',figsize=(15,10),title='Variacion de precios promedios a traves del tiempo')

ax.set_xlabel('Año y Mes')

ax.set_ylabel('Precio');

#MISMO GRAFICO PERO CON LA OFERTA

ax=precioxmes.plot(kind='bar',x='aniomes',y='cantidad',figsize=(15,10),title='Cantidad de publicaciones a traves del tiempo')

ax.set_xlabel('Año y Mes')

ax.set_ylabel('Cantidad de Publicaciones');
#SE VERIFICA UNA CANTIDAD MUY ELEVADA CON RESPECTO AL RESTO EN DIC-2016

aux=df[(df['anio']==2016) & ((df['mes']==12))].groupby(df.fecha.dt.day).size().reset_index().drop('fecha',axis=1)

aux=aux.reset_index()

aux['index']=aux['index']+1

aux.set_index('index',inplace=True)

ax=aux.plot(kind='bar',figsize=(15,10),title='Cantidad de publicaciones Diciembre-2016',legend=False)

ax.set_xlabel('Diciembre-2016')

ax.set_ylabel('Cantidad de Publicaciones');
precioxmes.sort_values(by='cantidad',ascending=False).head(3)
aux1=df[(df['anio']==2016) & (df['mes']==12)]

aux1=aux1.groupby(aux1.fecha.dt.day).size().reset_index()

aux1.columns=['fecha','2016-12']

aux2=df[(df['anio']==2016) & (df['mes']==10)]

aux2=aux2.groupby(aux2.fecha.dt.day).size().reset_index()

aux2.columns=['fecha','2016-10']

aux3=df[(df['anio']==2016) & (df['mes']==6)]

aux3=aux3.groupby(aux3.fecha.dt.day).size().reset_index()

aux3.columns=['fecha','2016-06']
aux=(aux1.merge(aux2,on='fecha',how='left'))#.merge(aux3,'fecha')

aux=aux.merge(aux3,on='fecha',how='left').set_index('fecha').fillna(0)

aux.head(32)
aux3.head()
ax=aux.plot(kind='bar',figsize=(15,10),title='Top 3 de meses con mas publicaciones',legend=True)

ax.set_xlabel('Dias del mes con publicaciones')

ax.set_ylabel('Cantidad de Publicaciones');
sns.set()

ax=aux.plot(kind='bar',stacked=True,figsize=(15,10),title='Top 3 de meses con mas publicaciones');

ax.set_xlabel('Dias del mes con publicaciones')

ax.set_ylabel('Cantidad de Publicaciones');
aux.tail()
df_top3=df[(df['provincia']=='Distrito Federal') | (df['provincia']=='Edo. de México') | (df['provincia']=='Jalisco')]
df.shape
df_top3.shape
df_top3.head(3)
#prom_ant=df_top3.groupby('provincia').antiguedad.mean().reset_index()

prom_ant=df_top3.groupby('provincia').agg({'antiguedad':'mean','precio':'mean','anio':'size'}).reset_index()

#    antiguedad.mean().reset_index()

prom_ant.columns=['Provincia','Antiguedad','Precio','Cantidad']

prom_ant.set_index('Provincia',inplace=True)

prom_ant
#merged_top3.plot(kind='bar')



prom_ant.plot(kind='bar',grid=True,subplots=True,sharex=True,figsize=(15,10));