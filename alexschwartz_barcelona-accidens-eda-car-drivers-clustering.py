import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
pd.set_option('display.max_columns',50)
sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
import chardet



#Funcion para predecir el encoding de un archivo:

def predict_encoding(file_path, n_lines=20):

        with open(file_path, 'rb') as f:

            rawdata = b''.join([f.readline() for _ in range(n_lines)])

            return chardet.detect(rawdata)['encoding'] 
file="../input/barcelona-accidents/2010_accidents.csv"

df_10=pd.read_csv(file,encoding=predict_encoding(file))
file='../input/barcelona-accidents/2011_accidents.csv'

df_11=pd.read_csv(file,encoding=predict_encoding(file))
file='../input/barcelona-accidents/2012_accidents.csv'

df_12=pd.read_csv(file,encoding=predict_encoding(file))
file='../input/barcelona-accidents/2013_accidents.csv'

df_13=pd.read_csv(file,encoding=predict_encoding(file))
predict_encoding('../input/barcelona-accidents/2014_accidents.csv')
#latin-1 es similar a W-1252. de todas formas veremos que sigue habiendo errores en palabras con caracteres particulares del Catalan

df_14=pd.read_csv('../input/barcelona-accidents/2014_accidents.csv',encoding='latin-1')

df_14.head()
file='../input/barcelona-accidents/2015_accidents.csv'

df_15=pd.read_csv(file,encoding=predict_encoding(file))
file='../input/barcelona-accidents/2016_accidents.csv'

df_16=pd.read_csv(file,encoding=predict_encoding(file))
#Reviso si los datasets contienen las mismas columnas:

print(df_10.columns==df_11.columns)

print(df_11.columns==df_12.columns)

print(df_12.columns==df_13.columns)
#cuenta de cantidad de columnas:

print('Features 2013:',df_13.columns.value_counts().sum())

print('Features 2014:',df_14.columns.value_counts().sum())

print('Features 2015:',df_15.columns.value_counts().sum())

print('Features 2016:',df_16.columns.value_counts().sum())

#similaridad de columas 14 y 15:

df_14.columns==df_15.columns
#Columnas distintas entre 14 y 15

set(df_14.columns).symmetric_difference(set(df_15.columns))
#columnas distintas entre 15 y 16

set(df_15.columns).symmetric_difference(set(df_16.columns))
#columnas similares entre 15 y 16

set(df_15.columns).intersection(set(df_16.columns))
#listado de columnas del 15:

df_15.columns
#Columnas distintas entre 13 y 16

set(df_13.columns).symmetric_difference(set(df_16.columns))
df_13.head()
df_13['Descripció torn'].unique()
torn_set=df_13.groupby(['Descripció torn','Hora de dia'])['Hora de dia'].mean()

torn_set
#creo feature para el dataset año 15 - Descrpició torn

def torn(hora):

    torn='Tarda'

    if hora>22 or hora<6:

        torn='Nit'

    elif hora <13:

        torn='Matí'

    return torn





df_15['Descripció torn']=df_15['Hora de dia'].map(lambda x:torn(x))
data_sin14=pd.concat([df_10,df_11,df_12,df_13,df_15,df_16])

data_sin14.shape
#reviso valores nulos:

data_sin14.isna().sum()
df_14.shape
#primero creo torn:



df_14['Descripció torn']=df_14['Hora de dia'].map(lambda x:torn(x))

df_14['Descripció torn'].unique()
df_14.columns
df_15.columns
#cambio nombre de columnas al año 14



df_14.columns=df_15.columns

df_14.head()
#creo tabla unica de codigos de distrito

table_dist=df_15.groupby(['Codi districte','Nom districte'])['Dia de mes'].mean().reset_index()

table_dist=table_dist.drop('Dia de mes',axis=1).set_index('Codi districte')

table_dist
#veo que los codigos sean los mismos

check_codi_14=df_14.groupby(['Codi districte','Nom districte'])['Dia de mes'].mean().reset_index()

check_codi_14
#reemplazo Descoengut por -1 en codigo y paso a int64

df_14['Codi districte']=df_14['Codi districte'].replace('Desconegut',-1)

df_14['Codi districte']=df_14['Codi districte'].astype(int)
#data frame intermedio

df_14_codi_dist=df_14.iloc[:,1:3]

#merge con nuevo nom districte

merged_codi_dist=df_14_codi_dist.merge(table_dist,on='Codi districte',how='left')

merged_codi_dist.isna().sum()
#reemplazo Nom distrticte

df_14['Nom districte']=merged_codi_dist['Nom districte_y']
#mismo procedimiento:

table_barri=df_15.groupby(['Codi barri','Nom barri'])['Dia de mes'].mean().reset_index()

table_barri=table_barri.drop('Dia de mes',axis=1).set_index('Codi barri')

table_barri
df_14['Codi barri']=df_14['Codi barri'].replace('Desconegut',-1)

df_14['Codi barri']=df_14['Codi barri'].astype(int)

df_14_codi_barri=df_14.iloc[:,3:5]

merged_codi_barri=df_14_codi_barri.merge(table_barri,on='Codi barri',how='left')

merged_codi_barri
merged_codi_barri.isna().sum()
df_14['Nom barri']=merged_codi_barri['Nom barri_y']
#mismo procedimiento - utilizo dataset sin 14 para tener mas calles

table_carrer=data_sin14.groupby(['Codi carrer','Nom carrer'])['Dia de mes'].count().reset_index()

table_carrer
repetidos=pd.DataFrame(table_carrer['Codi carrer'].value_counts())

repetidos[repetidos['Codi carrer']>1]
table_carrer[table_carrer['Codi carrer'].isin(['701353','189202','79907','701449'])]
#arreglo repetidos en la base general sin año 14

data_sin14['Nom carrer']=data_sin14['Nom carrer'].replace('Císter','Cister')

data_sin14['Nom carrer']=data_sin14['Nom carrer'].replace('Magalhães','Magalhaes')

data_sin14['Nom carrer']=data_sin14['Nom carrer'].replace('Josep M. Lladó','Josep Maria Lladó')

data_sin14['Nom carrer']=data_sin14['Nom carrer'].replace('Josep-Narcís Roca','Josep Narcís Roca')

reemp_desco=table_carrer[table_carrer['Codi carrer']==-1]

reemp_desco
#arreglo los Desconegut

data_sin14['Nom carrer']=data_sin14['Nom carrer'].replace([reemp_desco['Nom carrer']],'Desconegut')
table_carrer2=data_sin14.groupby(['Codi carrer','Nom carrer'])['Dia de mes'].count().reset_index()

table_carrer2
repetidos2=pd.DataFrame(table_carrer2['Codi carrer'].value_counts())

repetidos2[repetidos2['Codi carrer']>1]
#table carrer final:



table_carrer_f=data_sin14.groupby(['Codi carrer','Nom carrer'])['Dia de mes'].count().reset_index()

table_carrer_f=table_carrer_f.drop('Dia de mes',axis=1).set_index('Codi carrer')

table_carrer_f=pd.DataFrame(table_carrer_f)

table_carrer_f
df_14['Codi carrer']=df_14['Codi carrer'].replace('Desconegut',-1)

df_14['Codi carrer']=df_14['Codi carrer'].astype(int)

df_14_codi_carrer=df_14.iloc[:,5:7]

merged_codi_carrer=df_14_codi_carrer.merge(table_carrer_f,on='Codi carrer',how='left')

merged_codi_carrer
df_14['Nom carrer']=merged_codi_carrer['Nom carrer_y']
#Dia de la semana:

np.sort(df_14['Descripció dia setmana'].unique())==np.sort(df_15['Descripció dia setmana'].unique())

#Dia semana:

np.sort(df_14['Dia setmana'].unique())==np.sort(df_15['Dia setmana'].unique())
#Hora de dia

np.sort(df_14['Hora de dia'].unique())==np.sort(df_15['Hora de dia'].unique())
#Causa vianant:

np.sort(df_14['Descripció causa vianant'].unique())==np.sort(df_15['Descripció causa vianant'].unique())
df_14['Descripció causa vianant'].unique()
df_14['Descripció causa vianant']=df_14['Descripció causa vianant'].replace('No \x82s causa del  vianant','No és causa del  vianant')

df_14['Descripció causa vianant']=df_14['Descripció causa vianant'].replace('Transitar a peu per la cal\x87ada','Transitar a peu per la calçada')

df_14['Descripció causa vianant']=df_14['Descripció causa vianant'].replace('Desobeir el senyal del sem\x85for','Desobeir el senyal del semàfor')

np.sort(df_14['Descripció causa vianant'].unique())==np.sort(df_15['Descripció causa vianant'].unique())
#Tipo de vehicle implicat:

np.sort(df_14['Desc. Tipus vehicle implicat'].unique())
np.sort(df_15['Desc. Tipus vehicle implicat'].unique())
df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Autob£s','Autobús')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Autob£s articulado','Autobús articulado')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Cami¢n <= 3,5 Tm','Camión <= 3,5 Tm')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Cami¢n > 3,5 Tm','Camión > 3,5 Tm')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Otros veh¡c. a motor','Otros vehíc. a motor')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Tractocami¢n','Tractocamión')

df_14['Desc. Tipus vehicle implicat']=df_14['Desc. Tipus vehicle implicat'].replace('Tranv¡a o tren','Tranvía o tren')
np.sort(df_14['Desc. Tipus vehicle implicat'].unique())
#sexe

print(np.sort(df_14['Descripció sexe'].unique()))

print(np.sort(df_15['Descripció sexe'].unique()))
#tipus persona

np.sort(df_14['Descripció tipus persona'].unique())==np.sort(df_15['Descripció tipus persona'].unique())
#Descripcó victimizació

np.sort(df_14['Descripció victimització'].unique())==np.sort(df_15['Descripció victimització'].unique())
data_sin14.shape
df_14.shape
data_f=pd.concat([df_10,df_11,df_12,df_13,df_14,df_15,df_16])

data_f.shape
#Los horarios de los Años 14 y 15 estan de 0a12 horas, y no tienen descripción de Turno.



pd.pivot_table(data_f,index=['NK Any'],columns=['Hora de dia'],values='Mes de any',aggfunc=np.mean)
#genero una copia para trabajar

data=data_f.copy()
print(data.shape)

data.head()
#Reviso formato de los campos

data.info()
#saco espacios y mayúsculas de los nombres de las columnas:

data.rename(columns=lambda x: x.replace(' ','_').lower(),inplace=True)

data.columns
#set expendiente as id & index

data['id']=data["número_d'expedient"]

data.set_index('id',inplace=True)

data=data.drop("número_d'expedient",axis=1)

data.head()
#revisar cuantos stros dobles hay

data.index.value_counts()
#son repetidos?

data.duplicated().sum()
data[data.duplicated()]
data[data.index=='2011S0017']
data.index[0]
#Id has spaces after the number

data=data.reset_index()

data.id=data.id.apply(lambda x: x.strip())

data.id[0]
data=data.set_index('id')

data.loc['2010S000001']
data.loc['2016S006967']
pd.pivot_table(data,index=data.index,columns=['descripció_tipus_persona'],values='codi_barri')
data['coordenada_utm_(x)'].min(),data.lat.min()
#reemplazo Desconegut y paso a numerical la variable

data['coordenada_utm_(x)']=data['coordenada_utm_(x)'].apply(lambda x: x.replace('Desconegut','-1'))

data['coordenada_utm_(x)']=data['coordenada_utm_(x)'].apply(lambda x: x.replace(',','.')).astype(float)

data['coordenada_utm_(x)'].max(),data['coordenada_utm_(x)'].min()
#reemplazo Desconegut y paso a numerical la variable

data['coordenada_utm_(y)']=data['coordenada_utm_(y)'].apply(lambda x: x.replace('Desconegut','-1'))

data['coordenada_utm_(y)']=data['coordenada_utm_(y)'].apply(lambda x: x.replace(',','.')).astype(float)

data['coordenada_utm_(y)'].max(),data['coordenada_utm_(y)'].min()
from pyproj import Proj,transform



#prueba para encontrar codigos



#inProj = Proj(init='epsg:25830')

inProj = Proj(proj='utm',zone='31')

outProj = Proj(init='epsg:4326')

x1,y1 = 424249.09,4586527.04

x2,y2 = transform(inProj,outProj,x1,y1)

print(x2,y2)
#Function para cambiar de UTM a LatLong

#x1=utm(x) , y1=utm(y)

#output: y2=lat x2=long

def transform_latlong(x1,y1):

    inProj = Proj(proj='utm',zone='31')

    outProj = Proj(init='epsg:4326')

    x2,y2=transform(inProj,outProj,x1,y1)

    return(y2,x2)
def transform_latlong_tup(tup):

    inProj = Proj(proj='utm',zone='31')

    outProj = Proj(init='epsg:4326')

    x2,y2=transform(inProj,outProj,tup[0],tup[1])

    return (y2,x2)
#Creo columna con ambas coordenadas UTM

data['UTM_tuple']=data[['coordenada_utm_(x)','coordenada_utm_(y)']].apply(tuple,axis=1)

data.head()
#Creo listas con lat long transformada:

lat=[]

long=[]

for item in data.iloc[:,27]:

    a,b=transform_latlong_tup(item)

    lat.append(a)

    long.append(b)
#añado lat y long a todos los años con codigo cartográfico

data['geo_lat']=lat

data['geo_long']=long

data.head()
#cantidad de siniestros a por año

ax=sns.countplot('nk_any',data=data)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Año')

ax.set_title('Accidentes por año')
#Tipos de vehículos:

tipo_vehic=data.groupby('desc._tipus_vehicle_implicat').nk_any.count().sort_values(ascending=False)

tipo_vehic
#peso de la variable

(tipo_vehic/tipo_vehic.sum()).round(2)
#Feature engenieering, Otros:Grupos con menos del 3% de participacion.



def tipo_vehic(vehiculo):

    vehic_name=vehiculo

    vehic_list=['Motocicleta','Turismo','Ciclomotor','Bicicleta','Autobús','Taxi','Furgoneta']

    if not vehic_name in vehic_list:

            vehic_name='Otros'

    return vehic_name

data['tipo_vehic']=data['desc._tipus_vehicle_implicat'].apply(lambda x: tipo_vehic(x))

data.groupby('tipo_vehic').nk_any.count().sort_values(ascending=False)

#como funciona la siniestralidad por año y tipo de vehiculo. En ppio deberia esatr creciendo las bicicletas, debido a que la ciudad implementó bicing

vehic_year=data.groupby(['tipo_vehic','nk_any']).codi_barri.count().reset_index()

#pd.crosstab()

pd.pivot_table(vehic_year,index=['tipo_vehic'],columns=['nk_any'],values='codi_barri')
#tipo de implicado vision general:

tip_persona=data.groupby('descripció_tipus_persona').nk_any.count().reset_index()

tip_persona['perc']=(tip_persona.nk_any/tip_persona.nk_any.sum()).round(2)

tip_persona
f,ax=plt.subplots(1,2,figsize=(18,8))

data['descripció_tipus_persona'].value_counts().plot.pie(ax=ax[0])

ax[0].set_title('Tipo Persona')

ax[0].set_ylabel('')

sns.countplot('descripció_tipus_persona',data=data,ax=ax[1])

ax[1].set_title('Tipos persona')

plt.show()
#Por tipo de implicado:

pd.crosstab(data['tipo_vehic'],data.descripció_tipus_persona,normalize='index').round(2).plot(kind='bar')
#Sexo

sex=data.groupby('descripció_sexe').nk_any.count().sort_values(ascending=False).reset_index()

sex['perc']=(sex.nk_any/sex.nk_any.sum()).round(2)

sex
f,ax=plt.subplots(1,2,figsize=(18,8))

data['descripció_sexe'].value_counts().plot.pie(ax=ax[0])

ax[0].set_title('Sexe')

ax[0].set_ylabel('')

sns.countplot('descripció_sexe',data=data,ax=ax[1])

ax[1].set_title('Sexe')

plt.show()
#Elimino 5 registros con Sexo desconocido.

data=data[data['descripció_sexe']!='Desconegut']
#Veamos si hay alguna tendencia respecto al tipo de vehiculo:

pd.crosstab(data['tipo_vehic'],data.descripció_sexe,normalize='index').round(2).plot(kind='bar').set_xlabel('Tipo de Vehiculo')
pd.crosstab(data['descripció_tipus_persona'],data.descripció_sexe,normalize='index').round(2).plot(kind='bar').set_xlabel('Tipo de persona')
#Veamos edades:

data.edat.replace('Desconegut','-1',inplace=True)

data.edat=data.edat.astype(int)
#cuantas edades desconocidas hay?

data.edat.value_counts()[-1]
data.edat.mean()

sns.distplot(data['edat'],bins=30)
print('Male mu,std:',data[data['descripció_sexe']=='Home']['edat'].mean(),data[data['descripció_sexe']=='Home']['edat'].std())

print('Male mu,std:',data[data['descripció_sexe']=='Dona']['edat'].mean(),data[data['descripció_sexe']=='Dona']['edat'].std())





sns.distplot(data[data['descripció_sexe']=='Home']['edat'],bins=30,color='b')

sns.distplot(data[data['descripció_sexe']=='Dona']['edat'],bins=30,color='g')

#Edades por tipo de vehículo:

data.groupby(['tipo_vehic'])['edat'].mean().round(0).plot(kind='bar').set_title('Edad x Tipo Vehiculo')
#cuanto pesan las sin edad?

edad_desc=data[data['edat']==-1]

edad_desc.tipo_vehic.value_counts()/data.tipo_vehic.value_counts()
#reemplazo las edades desconocidas por la media de cada tipo de vehiculo.

edad_media=np.round(data.edat.mean(),0).astype(int)



def replace_age(cols):

    age=cols[0]

    vehic=cols[1]

    

    if age == -1:

        if vehic=='Autobús':

            return 58

        elif vehic=='Otros':

            return 46

        elif vehic=='Taxi':

            return 42

        elif vehic == 'Furgoneta':

            return 43

        elif vehic == 'Bicicleta':

            return 30

        else:

            return edad_media

    else:

        return age



    

data.edat=data[['edat','tipo_vehic']].apply(replace_age,axis=1)
#edades por tipo de vehiculo y sexo:

pd.pivot_table(data,values=['edat'],index=['tipo_vehic'],columns=['descripció_sexe'],aggfunc=np.mean).round(2).plot(kind='bar').set_title('Edad x Tipo de Vehiculo y sexo')
#edades por tipo de vehiculo y tipo de persona

pd.pivot_table(data,values=['edat'],index=['tipo_vehic'],columns=['descripció_tipus_persona'],aggfunc=np.mean).round(2).plot(kind='bar',figsize=(10,6)).set_title('Edad por Tipo Vehic y Persona')
data.groupby(['descripció_tipus_persona']).edat.mean().round(2)
#Severidad

sev=data.groupby('descripció_victimització').nk_any.count().sort_values(ascending=False).reset_index()

sev['perc']=(sev.nk_any/sev.nk_any.sum()).round(2)

sev

#veamos accidentes por tipo de vehículo:



pd.crosstab(data['tipo_vehic'],data.descripció_victimització,normalize='index').round(2)
#por tipo de persona?

pd.crosstab(data['descripció_tipus_persona'],data.descripció_victimització,normalize='index').round(2)#.plot(kind='bar').set_xlabel('Tipo de persona')
#los graves por tipo de vehiculo?

graves=data[data['descripció_victimització']=='Mort']

graves.tipo_vehic.value_counts()
graves.descripció_tipus_persona.value_counts()
graves.groupby(['tipo_vehic','descripció_tipus_persona']).nk_any.count().unstack()
#cantidad de siniestros a por año

ax=sns.countplot('nk_any',data=data)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Año')

ax.set_title('Accidentes por año')
data.groupby(['tipo_vehic','nk_any']).codi_barri.count().unstack().T.plot(figsize=(10,5)).set_title('Accidentes por Tipo de vehic y Año')
#MES

ax=sns.countplot('mes_de_any',data=data)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Nr Mes')

ax.set_title('Accidentes por mes del año')
#Día del mes

ax=sns.countplot('dia_de_mes',data=data)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Dia del Mes')

ax.set_title('Accidentes por día del mes')

plt.xticks(rotation=90)
#Día de la semana

ax=sns.countplot('descripció_dia_setmana',data=data,order=['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',

                                        'Divendres', 'Dissabte', 'Diumenge'])

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Dia de la semana')

ax.set_title('Accidentes por dia de la semana')

plt.xticks(rotation=90)
data2016=data[data['nk_any']==2016]
heat_data=data2016.groupby(['descripció_dia_setmana', 'hora_de_dia'])['nk_any'].count().to_frame().unstack()

heat_data.columns = heat_data.columns.droplevel()

heat_data = heat_data.reindex(index = ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',

                                        'Divendres', 'Dissabte', 'Diumenge'])

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

plt.figure(figsize=(15,5))

sns.heatmap(heat_data,linewidths=.2,cmap=cmap)

plt.title('Accidents per dia de la setmana y hora 2016', fontsize=25)
#paso a numerical el dia de la semana

def dia_semana(dia):

    if dia=='Diumenge':

        return 1

    elif dia=='Dilluns':

        return 2

    elif dia=='Dimarts':

        return 3

    elif dia=='Dimecres':

        return 4

    elif dia=='Dijous':

        return 5

    elif dia=='Divendres':

        return 6

    else:

        return 7

data['num_dia_semana']=data.descripció_dia_setmana.apply(lambda x: dia_semana(x))
data.groupby(['tipo_vehic','num_dia_semana']).nk_any.count().unstack().T.plot(figsize=(10,5))
#Distritos:

x=data.nom_districte.value_counts().index

y=data.nom_districte.value_counts()

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Distrito')

ax.set_title('Accidentes segun Distrito')

plt.xticks(rotation=90)
#Accidentes por nombre de distrito y tipo de vehiculo

dist_vehic=data.groupby(['nom_districte','tipo_vehic']).nom_barri.count().unstack()

sns.heatmap(dist_vehic,linewidths=.2,cmap=cmap)

plt.title('Accidentes x Distrito y tipo de vehiculo')
#Peso de distrito por tipo de vehiculo

dist_vehic2=pd.crosstab(data['tipo_vehic'],data.nom_districte,normalize='index').round(2)

sns.heatmap(dist_vehic2,linewidths=.2,cmap=cmap)

plt.title('Importancia del distrito en cada Tipo de vehiculo')
#BARRIOS



data['barri_dist']=data['nom_barri']+' - '+data['nom_districte']



x=data.barri_dist.value_counts().head(10).index

y=data.barri_dist.value_counts().head(10)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Distrito')

ax.set_title('Accidentes x Barrio')

plt.xticks(rotation=90)
#Calles



x=data.nom_carrer.value_counts().head(10).index

y=data.nom_carrer.value_counts().head(10)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Distrito')

ax.set_title('Accidentes por Calles')

plt.xticks(rotation=90)
data['calle_barri']=data['nom_carrer']+' - '+data['nom_barri']



x=data.calle_barri.value_counts().head(10).index

y=data.calle_barri.value_counts().head(10)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Distrito')

ax.set_title('Accidentes por Calle-Barrio')

plt.xticks(rotation=90)
turismo=data[data['tipo_vehic']=='Turismo']

turismo.shape
#Siniestros por año:

ax=sns.countplot('nk_any',data=turismo)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Año')

ax.set_title('Accidentes de autos por año')
#composicion entre tipo de persona:

f,ax=plt.subplots(1,2,figsize=(18,8))

turismo['descripció_tipus_persona'].value_counts().plot.pie(ax=ax[0])

ax[0].set_title('Tipo Persona')

ax[0].set_ylabel('')

sns.countplot('descripció_tipus_persona',data=turismo,ax=ax[1])

ax[1].set_title('Tipos persona')

plt.show()
tipo_persona=turismo.descripció_tipus_persona.value_counts().reset_index()

tipo_persona['perc']=(tipo_persona.descripció_tipus_persona/tipo_persona.descripció_tipus_persona.sum()).round(2)

tipo_persona
turismo.groupby(['descripció_tipus_persona','nk_any']).codi_barri.count().unstack().T.plot().legend(loc='lower left')
#Creo dataset de Conductores de autos

conduc=turismo[turismo['descripció_tipus_persona']=='Conductor']

conduc.shape
# vamos a ver sexo:



conduc.descripció_sexe.value_counts(normalize=True).round(2).plot(kind='pie')
#Edades:

np.round(conduc.edat.mean(),2),np.round(conduc.edat.std(),2)

print('Male mu,std:',np.round(conduc[conduc['descripció_sexe']=='Home']['edat'].mean(),2),np.round(conduc[conduc['descripció_sexe']=='Home']['edat'].std(),2))

print('Male mu,std:',np.round(conduc[conduc['descripció_sexe']=='Dona']['edat'].mean(),2),np.round(conduc[conduc['descripció_sexe']=='Dona']['edat'].std(),2))





sns.distplot(conduc[conduc['descripció_sexe']=='Home']['edat'],bins=30,color='b')

sns.distplot(conduc[conduc['descripció_sexe']=='Dona']['edat'],bins=30,color='g')

sns.distplot(data['edat'],bins=30,color='r')
g = sns.factorplot( "descripció_sexe",'edat', data=conduc, kind="box")

g.set_axis_labels("Sexo", "Edad")
#Quantiles de edad para resumir la distribución en categorias de edades:

print('IQ:',conduc.edat.quantile(0.25))

print('IIQ:',conduc.edat.quantile(0.5))

print('IIIQ:',conduc.edat.quantile(0.75))

def rango_edad(edad):

    if edad<28:

        return '0-27'

    elif edad<36:

        return '28-35'

    elif edad<47:

        return '36-46'

    else:

        return '47-'
conduc['rango_edad']=conduc.edat.apply(lambda x: rango_edad(x))

conduc.rango_edad.value_counts()
#Revisar Severidad:

conduc.descripció_victimització.value_counts()
conduc.groupby(['descripció_victimització','rango_edad']).nk_any.count().unstack()
#MES

ax=sns.countplot('mes_de_any',data=conduc)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Nr Mes')

ax.set_title('Accidentes x Mes')
#Día del mes

ax=sns.countplot('dia_de_mes',data=conduc)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Dia del Mes')

ax.set_title('Accidentes x Día del Mes')

plt.xticks(rotation=90)
#Día de la semana

ax=sns.countplot('descripció_dia_setmana',data=conduc,order=['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',

                                        'Divendres', 'Dissabte', 'Diumenge'])

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Dia de la semana')

plt.xticks(rotation=90)
#dataframe de conductores de autos accidentes 2016

conduc2016=conduc[conduc['nk_any']==2016]
# Number of accident per hour

accidents_hour = conduc2016.hora_de_dia.value_counts().sort_index()



# plot accidents per hour

accidents_hour.plot(kind='bar',figsize=(12,7), color='orange', alpha=0.5)



# title and x,y labels

plt.title('Accidents in Barcelona in 2016', fontsize=20)

plt.xlabel('Hour',fontsize=16)

plt.ylabel('Number of accidents',fontsize=16);
heat_data=conduc2016.groupby(['descripció_dia_setmana', 'hora_de_dia'])['nk_any'].count().to_frame().unstack()

heat_data.columns = heat_data.columns.droplevel()

heat_data = heat_data.reindex(index = ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',

                                        'Divendres', 'Dissabte', 'Diumenge'])

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

plt.figure(figsize=(15,6))

sns.heatmap(heat_data,linewidths=.2,cmap=cmap)

plt.title('Accidents per dia de la setmana y hora 2016', fontsize=25)
#Distritos:

x=conduc.nom_districte.value_counts().index

y=conduc.nom_districte.value_counts()

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Distrito')

ax.set_title('Accidentes por Districte')

plt.xticks(rotation=90)
dist_vehic=conduc.groupby(['nom_districte','descripció_dia_setmana']).nom_barri.count().unstack()

sns.heatmap(dist_vehic,linewidths=.2,cmap=cmap)
#BARRIOS

data['barri_dist']=data['nom_barri']+' - '+data['nom_districte']



x=data.barri_dist.value_counts().head(15).index

y=data.barri_dist.value_counts().head(15)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('BARRIO - DISTRITO')

ax.set_title('Accidents per Barri-Districte')

plt.xticks(rotation=90)
plt.figure(figsize=(10,5))



plt.subplot(1, 2, 1)

x=conduc.nom_carrer.value_counts().head(15).index

y=conduc.nom_carrer.value_counts().head(15)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Calle')

plt.xticks(rotation=90)



plt.subplot(1, 2, 2)

x=conduc.calle_barri.value_counts().head(15).index

y=conduc.calle_barri.value_counts().head(15)

ax=sns.barplot(x=x,y=y)

ax.set_ylabel('#Accidentes')

ax.set_xlabel('Calle - Barrio')

plt.xticks(rotation=90)
import folium

from folium.plugins import HeatMap,Fullscreen



#creo mapa de fondo - situado en Barcelona

barcelona_map2 = folium.Map(location=[41.38879, 2.15899], zoom_start=13)



#dataframe con lat y long

heat_df=conduc[['geo_lat','geo_long']]



heat_data=[[row['geo_lat'],row['geo_long']] for index,row in heat_df.iterrows() ]



#agrego el mapa de calor al mapa base

HeatMap(heat_data).add_to(barcelona_map2)



#agrego funcionalidad para poder ponerlo en full screen

Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(barcelona_map2)



print('Mapa de Calor Accidentes de conductores de autos año 2016')



barcelona_map2
barcelona_map3 = folium.Map(location=[41.38879, 2.15899], zoom_start=13)



heat_df=conduc2016[['geo_lat','geo_long']]



heat_data=[[row['geo_lat'],row['geo_long']] for index,row in heat_df.iterrows() ]



HeatMap(heat_data).add_to(barcelona_map3)



Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(barcelona_map3)



print('Mapa de Calor Accidentes de Conductores de Auto año 2016')

barcelona_map3
#prepare de dataset for KMeans:

conduc_cluster16=conduc2016[['descripció_sexe','codi_barri','codi_districte','codi_carrer','dia_de_mes','num_dia_semana','edat','mes_de_any','hora_de_dia']]
#Formato de cada instancia

conduc_cluster16.info()
#Creando OneHot Encoding for categorical data

conduc_cluster16=pd.get_dummies(conduc_cluster16,columns=['descripció_sexe'],drop_first=True)

conduc_cluster16.head()
from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV



parameters={'init':['k-means++','random'],

           'n_init':[10,100],

           'tol':[0.1,0.01,0.001,0.0001,0.00001],

           'n_clusters': np.arange(1,21)}









grid_search=GridSearchCV(KMeans(random_state=42),parameters,cv=5, verbose=True)

grid_search.fit(conduc_cluster16)

#mejores parametros de la grilla

grid_search.best_params_
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler



base_scaled=StandardScaler().fit_transform(conduc_cluster16)



distortions = []

for i in range(1, 20):

    

    

    km = KMeans(

        n_clusters=i, init='k-means++',

        n_init=10, max_iter=300,

        tol=0.01, random_state=42

    )

    km.fit(base_scaled)

    distortions.append(km.inertia_)



# plot

plt.plot(range(1,20), distortions, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.show()
from sklearn.cluster import KMeans



km = KMeans(

        n_clusters=10, init='k-means++',

        n_init=10, max_iter=300,

        tol=0.01, random_state=42)



km.fit(base_scaled)

conduc2016['cluster1']=km.labels_
#Check de tener creado la variable con los clusters asignados

conduc2016.head()
conduc2016.cluster1.unique()
import folium

from folium.plugins import Fullscreen





barcelona_map = folium.Map(location=[41.38879, 2.15899], zoom_start=14)



colors=['red', 'blue', 'green', 'purple', 'orange', 'darkred',

             'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',

             'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',

             'gray', 'black', 'lightgray']





for i in range(0,len(conduc2016.cluster1)):

    color=colors[conduc2016.cluster1[i]]

    

    folium.Marker([conduc2016.geo_lat[i],conduc2016.geo_long[i]],

                 popup=f'cluster:{conduc2016.cluster1[i]}',

                 icon=folium.Icon(color)).add_to(barcelona_map)
Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

    ).add_to(barcelona_map)

    

barcelona_map
size_clusters=conduc2016.cluster1.value_counts(normalize=True).round(2).sort_index()

size_clusters.plot(kind='bar',color='blue',alpha=.8)



plt.title('Representatividad de cada Cluster', fontsize=20)

plt.xlabel('Cluster nmb',fontsize=16)

plt.ylabel('percentage',fontsize=16);
#Age



print('Edad media:',np.round(conduc2016.edat.mean(),2))



conduc2016.groupby(['cluster1']).edat.mean().round(2).plot(kind='bar',color='blue',alpha=.8).set_title('Edad media por Cluster')



#Age distribution

age_dist=conduc2016.groupby(['cluster1']).rango_edad.value_counts(normalize=True).unstack().round(2)

age_dist.plot(kind='bar',stacked=True).set_title('Distribucion por rango etareo de cada cluster')

g = sns.factorplot( "cluster1",'edat', data=conduc2016, kind="box")

g.set_axis_labels("Cluster", "Edad")
#Sexo

conduc2016.groupby(['cluster1']).descripció_sexe.value_counts(normalize=True).unstack().plot(kind='bar',stacked=True)

plt.title('Composición por Sexo en cada cluster')
#Distrito



cmap2 = sns.cubehelix_palette(start=2,light=1, as_cmap=True)



plt.figure(figsize=(15,6))



dist_clus_num=conduc2016.groupby(['cluster1']).nom_districte.value_counts(normalize=False).unstack()



sns.heatmap(dist_clus_num,linewidths=.2,cmap=cmap2).set_title('Accidentes sobre el total - clusters')

#DIAS

dia_clus=conduc2016.groupby(['cluster1']).descripció_dia_setmana.value_counts(normalize=False).unstack()

dia_clus=dia_clus.reindex(columns = ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',

                                        'Divendres', 'Dissabte', 'Diumenge'])

plt.figure(figsize=(15,6))

sns.heatmap(dia_clus,linewidths=.2,cmap=cmap2).set_title('Dias de accidentes - clusters')

#HORA



hora_clus=conduc2016.groupby(['cluster1']).hora_de_dia.value_counts().unstack()

#index=conduc2016.barri_dist.value_counts().sort_values(ascending=False).index

#barri_clus=barri_clus.reindex(index=index)

plt.figure(figsize=(10,6))

sns.heatmap(hora_clus,linewidths=.2,cmap=cmap2).set_title('Dias de accidentes - clusters')

#BARRIO

barri_clus=conduc2016.groupby(['cluster1']).barri_dist.value_counts().unstack().T

index=conduc2016.barri_dist.value_counts().sort_values(ascending=False).index

barri_clus=barri_clus.reindex(index=index)

plt.figure(figsize=(15,20))

sns.heatmap(barri_clus.head(20),linewidths=.2,cmap=cmap2).set_title('Dias de accidentes - clusters')

conduc2016.to_csv('Bcn_Accidents_2016_Clusters.txt')