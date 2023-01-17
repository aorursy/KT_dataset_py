# import this
import os
print(os.listdir("../input"))
import pandas as pd
datos = pd.read_csv('../input/historico-nombres.csv')
datos.shape
# tomamos una muestra más pequeña para avazar rápido
# datos.head()
# datos[1:5:2] 
muestra = datos.sample(10)
print(len(muestra))
print(muestra.shape)
# muesta vs print(muestra)
muestra.info()
# dir(muestra) 
# seteamos el anio como indice de los registros
# muestra, muestra['nombre'] , muestra[['nombre','cantidad']],muestra.index
# muestra.set_index('anio')
# verificamos la muestra
muestra
# ¿Qué pasó? no se guardo el cambio
muestra = muestra.set_index('anio')
# muestra = muestra.set_index('anio',inplace=True)
muestra
# muestra.loc[2010] # usando las etiquetas
muestra.iloc[3] # usando la posición
muestra['nombre']
# strip, split, replace, lower 
nombres = ( muestra['nombre'].str.strip()
                             .str.split(' ',expand=True) )
nombres
s1,s2,s3 = nombres[0], nombres[1].dropna(), nombres[2].dropna()
pd.concat([s1,s2,s3])
# nombres.melt()
# Indexación lógica
nombres = nombres[nombres.apply(len) > 0]
len(nombres)
s = nombres
# s.groupby(s).count()
s.groupby(level=0).count().plot(kind='bar') #ojo con el índice, tiene que ser el año
s.groupby(s).count().plot(kind='barh')
datos = pd.read_csv('../input/historico-nombres.csv')
# ¿Cómo encontramos la persona con más nombres ?
#datos['nombre'].str.strip().str.split(' ').apply(len).sort_values(ascending=False)
datos['nombre'].str.strip().dropna().str.split(' ').apply(len).sort_values(ascending=False)
# Vemos que está pasando con los que tienen demasiados nombres
# erroneos = 
datos.iloc[[8186346,1182489,3783178,232176,2047569,1896082,7869591,7076041,5585598,6877973]]['nombre'] #.values
erroneos.str.replace('\s+',' ')
# datos['nombre'].str.replace('\s+',' ').str.strip().dropna().str.split(' ').apply(len).sort_values(ascending=False)
datos.iloc[[34184,
1730337,
889444,
1386199,
3869074,
677216]]['nombre'].values
cantidad_de_nombres = datos['nombre'].str.replace('\s+',' ').str.strip().dropna().str.split(' ').apply(len)
cantidad_de_nombres
df = datos.loc[cantidad_de_nombres[cantidad_de_nombres < 3].sample(1000).index]
df = df['nombre'].str.replace(r'\s+',' ').str.strip()
df
dos_nombres = df.str.split(' ',expand=True).rename(columns={0:'primer',1:'segundo'}).dropna()
dos_nombres.groupby(by='primer').count().sort_values(by='segundo',ascending=False)
%matplotlib inline
import seaborn as sns
sns.set()
# datos.shape
datos = datos.sample(2000)
datos['nombre'] = datos['nombre'].str.strip().str.replace('\s+',' ').str.lower()
assert datos['nombre'].is_unique
# datos
# datos[datos['nombre'].duplicated()]
es_maria = datos['nombre'].str.contains('maria')#.sum()
datos[es_maria]
datos['long_nombre']=datos['nombre'].str.replace(' ','').apply(len)
# datos.head()
# sns.distplot(datos['long_nombre'],kde=False)
sns.countplot(datos['long_nombre'],color='Black')
cant_vs_long = datos[['cantidad','long_nombre']].groupby('long_nombre',as_index=False).sum()
sns.barplot(x='long_nombre',y='cantidad',data=cant_vs_long,color='Black')
# primer_nombre = 
datos['nombre'].str.split(' ',expand=True).drop(columns=[1,2,3,4]).dropna().rename(columns={0:'nombre'})
primer_nombre = primer_nombre.merge(datos[['cantidad']],right_index=True,left_index=True)
primer_nombre
# primer_nombre = 
primer_nombre.groupby('nombre',as_index=False).sum()
primer_nombre['inicial'] = primer_nombre['nombre'].str.slice(0,1)
# cant_iniciales =
primer_nombre[['inicial','cantidad']].groupby('inicial',as_index=False).sum()
sns.barplot(x='inicial',y='cantidad',data=cant_iniciales,color='black')
nombres = datos['nombre'].str.strip().str.split(' ',expand=True).drop(columns=[2,3,4]).dropna().rename(columns={0:'primer',1:'segundo'})
iniciales = nombres.applymap(lambda x: x[0])
iniciales['dummy'] = 1
iniciales = iniciales.groupby(by=['primer','segundo'],as_index=False).sum()
# iniciales
sns.heatmap(pd.crosstab(index=iniciales['primer'],columns=iniciales['segundo'],values=iniciales['dummy'],aggfunc=sum))