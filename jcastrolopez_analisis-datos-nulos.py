# importacion general de librerias y de visualizacion (matplotlib y seaborn)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as datetime

from math import pi

import sys





%matplotlib inline



plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib

#plt.rcParams['figure.figsize'] = (20, 10)



sns.set(style="whitegrid") # seteando tipo de grid en seaborn



pd.options.display.float_format = '{:20,.2f}'.format # suprimimos la notacion cientifica en los outputs
# Cargamos el dataframe

df = pd.read_csv('../input/mexican_zonaprop_datasets/train.csv',

        index_col='id',

        parse_dates=['fecha'])
#Creo columnas utiles a lo largo del analisis.

df['nulls'] = df.isnull().apply(np.sum,axis=1)

df['mes'] = df['fecha'].dt.month

df['ano'] = df['fecha'].dt.year
#Veamos con cuantos campos vacios nos encontramos:

nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['nulls'])

nulls['porcentaje'] = round(100*nulls['nulls'] / len(df), 2)

nulls=nulls[:15]

nulls.reset_index(inplace=True)

nulls.rename(columns={'index':'Campos'},inplace=True)

camposNull=nulls.Campos.tolist()

nulls
#Veamos que provincia es la que más datos omite:

def sumNuls (x):

    return x.isnull().sum()



dfNullTot = df.groupby('provincia').apply(sumNuls).apply(np.sum,axis=1).reset_index()

dfNullTot.rename(columns={0:'cantidad'}, inplace=True)

dfNullTot=dfNullTot.sort_values(by='cantidad', ascending=False)[0:15]



plt.figure(figsize=(15, 10))

g = sns.barplot(x=dfNullTot['cantidad'], y=dfNullTot['provincia'], orient='h')

g.set_title("Top 15 Provincias donde se omitieron datos", fontsize=16)

g.set_xlabel("Cantidad", fontsize=14)

g.set_ylabel("Provincia", fontsize=14)
#Veamos que provincia omitió más campos:



dfNullTot = df.groupby(['provincia','ciudad']).agg({'nulls':'sum'}).reset_index().sort_values(by='nulls', ascending=False)[0:15]

plt.figure(figsize=(15, 10))



g = sns.barplot(x=dfNullTot['nulls'], y=dfNullTot['ciudad'], orient='h', hue=dfNullTot['provincia'], dodge=False)

g.set_title("Top 15 ciudades donde se omitieron datos", fontsize=16)

g.set_xlabel("Cantidad de datos omitidos", fontsize=14)

g.set_ylabel("Ciudad", fontsize=14)
#Vamos a investigar el top 5 de las ciudades que mas datos omitieron para ver que categorias son las más comunes:



from collections import OrderedDict



def perform_stats(df, camposNull):

    # lo usamos para preservar el orden de insercion

    data = OrderedDict()

    

    for i in camposNull:

        data[i] = df[i].isnull().sum()



    return pd.Series(data)



dfCiudadesRadar = df.groupby('ciudad').agg({'nulls':'sum'}).sort_values(by='nulls',ascending = False)[0:5].reset_index()

print(dfCiudadesRadar.head(5))

ciudadesRadar = list(dfCiudadesRadar['ciudad'])

dfCiudadesRadar = df[df.ciudad.isin(ciudadesRadar)].groupby('ciudad').apply(perform_stats, camposNull)

dfCiudadesRadar.drop(columns=['ciudad','provincia','lng','lat'], inplace=True)
#Hagamos un radar Plot para comparar visualmente estas 5 ciudades



Attributes = list(dfCiudadesRadar)

AttNo = len(Attributes)



ax = plt.subplot(111, polar=True)



values = dfCiudadesRadar.iloc[0].tolist()

values += values [:1]



angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]

angles += angles [:1]



#Add the attribute labels to our axes

plt.xticks(angles[:-1],Attributes)

#Plot the line around the outside of the filled area, using the angles and values calculated before

ax.plot(angles,values)

#Fill in the area plotted in the last line

ax.fill(angles, values, 'teal', alpha=0)

#Give the plot a title and show it

plt.figtext(1,1,"Benito Juarez",color="teal")

#----------------------------------------------



values = dfCiudadesRadar.iloc[1].tolist()

values += values [:1]

#Plot the line around the outside of the filled area, using the angles and values calculated before

ax.plot(angles,values)

#Fill in the area plotted in the last line

ax.fill(angles, values, 'green', alpha=0)

#Give the plot a title and show it

plt.figtext(1,0.95,"Monterrey",color="green")

#----------------------------------------------



values = dfCiudadesRadar.iloc[2].tolist()

values += values [:1]

#Plot the line around the outside of the filled area, using the angles and values calculated before

ax.plot(angles,values)

#Fill in the area plotted in the last line

ax.fill(angles, values, 'blue', alpha=0)

#Give the plot a title and show it

plt.figtext(1,0.90,"Mérida",color="blue")



#----------------------------------------------

values = dfCiudadesRadar.iloc[3].tolist()

values += values [:1]

#Plot the line around the outside of the filled area, using the angles and values calculated before

ax.plot(angles,values)

#Fill in the area plotted in the last line

ax.fill(angles, values, 'red', alpha=0)

#Give the plot a title and show it

plt.figtext(1,0.85,"Querétaro",color="red")



#----------------------------------------------

values = dfCiudadesRadar.iloc[4].tolist()

values += values [:1]

#Plot the line around the outside of the filled area, using the angles and values calculated before

ax.plot(angles,values)

#Fill in the area plotted in the last line

ax.fill(angles, values, 'darkorchid', alpha=0)

#Give the plot a title and show it



plt.figtext(1,0.80,"San Luis Potosí",color="darkorchid")

ax.set_title('Campos vacíos de las 5 que más datos omitieron \n\n', fontdict={'fontsize':16})

plt.show()
#Queremos ver cantidad de tipos de propiedades en estas ciudades para tener una nocion de las cantidades

#Investiguemos de las ciudades que mas nulls tienen, los tipos de propiedades donde mas datos se omiten

dfNulsProp = df[df.ciudad.isin(ciudadesRadar)].groupby(['ciudad','tipodepropiedad']).agg({'nulls':'sum'})

dfNulsProp.reset_index(inplace=True)



newDf = pd.DataFrame(columns=dfNulsProp.columns)



for i in ciudadesRadar: 

     newDf = newDf.append(dfNulsProp[dfNulsProp.ciudad == i].sort_values(by="nulls", ascending=False)[0:4])



plt.figure(figsize=(10, 10))

g = sns.barplot(x=newDf['tipodepropiedad'], y=newDf['nulls'], hue=newDf['ciudad'])

g.set_title("Cantidad de propiedades en estas ciudades", fontsize=16)

g.set_xlabel("Tipo de Propiedad", fontsize=14)

g.set_ylabel("Cantidad de Nulos", fontsize=14)
#Vamos a hacer un heatmap con los tipos de propiedad como filas y como columnas los campos con más nulls.

#Cada celda va a tener un porcentaje de la cantidad de nulls que posee.

#Los datos se van a sacar de agrupar estas 5 ciudades que mas nulls omitieron ( son estas ciudades que estamos investigando)

dfHeat = df[df.ciudad.isin(ciudadesRadar)]

dfHeat.tipodepropiedad.value_counts()

dfHeat = dfHeat[dfHeat.antiguedad.isnull() | dfHeat.direccion.isnull() | dfHeat.metrostotales.isnull()

                | dfHeat.idzona.isnull() | dfHeat.banos.isnull() | dfHeat.habitaciones.isnull()]

dfHeat.tipodepropiedad.value_counts()



def nulsAntyDir(df):

    # lo usamos para preservar el orden de insercion

    data = OrderedDict()

    data['nul_antiguedad'] = df['antiguedad'].isnull().sum()

    data['nul_direccion'] = df['direccion'].isnull().sum()

    data['nul_metrostotales'] = df['metrostotales'].isnull().sum()

    data['nul_idzona'] = df['idzona'].isnull().sum()

    data['nul_banos'] = df['banos'].isnull().sum()

    data['nul_habitaciones'] = df['habitaciones'].isnull().sum()



    return pd.Series(data)

print(dfHeat.antiguedad.isnull().sum())

print(dfHeat.direccion.isnull().sum())

dfHeat = dfHeat.groupby('tipodepropiedad').apply(nulsAntyDir).sort_values(by='nul_direccion', ascending = False)[0:10]

dfHeat = dfHeat.reset_index()

tiposDePropiedadesHeat = list(dfHeat.index)

print(tiposDePropiedadesHeat)
#Cambiamos los valores de este data frame, para empezar a trabajar con valores porcentuales

dfHeatTotales = df[df.ciudad.isin(ciudadesRadar)]

dfHeatTotales = dfHeatTotales.groupby('tipodepropiedad').agg({'idzona':'size'}).reset_index()

dfHeatTotales = dfHeatTotales[dfHeatTotales.index.isin(tiposDePropiedadesHeat)]

dfHeatTotales = dfHeatTotales.rename(columns={'idzona':'cantidadTotal'})

dfHeatTotales=dfHeatTotales.merge(dfHeat)

dfHeatTotales.set_index('tipodepropiedad', inplace=True)





dfHeatTotales['Nulls antiguedad'] = dfHeatTotales.nul_antiguedad/dfHeatTotales.cantidadTotal

dfHeatTotales['Nulls direccion'] = dfHeatTotales.nul_direccion/dfHeatTotales.cantidadTotal

dfHeatTotales['Nulls metrostotales'] = dfHeatTotales.nul_metrostotales/dfHeatTotales.cantidadTotal

dfHeatTotales['Nulls idzona'] = dfHeatTotales.nul_idzona/dfHeatTotales.cantidadTotal

dfHeatTotales['Nulls banos'] = dfHeatTotales.nul_banos/dfHeatTotales.cantidadTotal

dfHeatTotales['Nulls habitaciones'] = dfHeatTotales.nul_habitaciones/dfHeatTotales.cantidadTotal

dfHeatTotales.drop(columns={'cantidadTotal','nul_antiguedad','nul_metrostotales','nul_direccion','nul_idzona','nul_banos','nul_habitaciones'}

                   , inplace=True)



dfHeatTotales
# Creamos el heatmap con el dataframe logrado

plt.figure(figsize=(15, 10))

cmap = sns.diverging_palette(200, 250, as_cmap=True)

ax=sns.heatmap(dfHeatTotales,

            cmap='Greens')

ax.set_facecolor('xkcd:grey')

ax.set_title('Porcentaje de nulos según tipo de propiedad \n', fontdict={'fontsize':20})

ax.set_xlabel('')

ax.set_ylabel('')
#Queremos ver la cantidad de nulls y la cantidad de publicaciones en total

dfAnualNulls = df.groupby('ano').agg({'nulls':'sum'}).reset_index()

dfTotPubl = df.groupby('ano').size().reset_index()

dfTotPubl.rename(columns={0:'total'}, inplace=True)



fig, ax = plt.subplots(figsize=( 15,10))

ax.plot(dfTotPubl['ano'], dfTotPubl['total'], label='Publicaciones', color='blue')

ax.plot(dfAnualNulls['ano'], dfAnualNulls['nulls'], label='Nulls', color='red')



plt.xticks([2012,2013,2014,2015,2016])

plt.grid(b=True, which='major', axis='both')

ax.set_xlabel("\n Año", fontsize=14)

ax.set_ylabel("Cantidad \n", fontsize=14)

ax.legend(loc='best')    

ax.set_title('Publicaciones vs Nulls por año \n', fontdict={'fontsize':16})
#Viendo el grafico anterior, nos preguntamos como se comportan las provincias ahora:

provincias = df.groupby('provincia').size().reset_index().provincia.tolist()



dfAnos = df.groupby(['provincia','ano']).agg({'nulls':'sum'}).reset_index()



dfAnos = dfAnos.merge(df.groupby(['provincia','ano']).size().reset_index())

dfAnos.rename(columns={0:'TotalPubl'}, inplace=True)



fig, ax = plt.subplots(figsize=( 15,10))

setLabels = 0

for i in provincias:

    label = ""



    if(setLabels == 0):

        ax.plot(dfAnos[dfAnos['provincia']== i ].ano, dfAnos[dfAnos['provincia']== i ].TotalPubl, color='blue', label='Publicaciones')

        ax.plot(dfAnos[dfAnos['provincia']== i ].ano, dfAnos[dfAnos['provincia']== i ].nulls, color='red',label='Campos No Completados')

        setLabels = 1

    else:

        ax.plot(dfAnos[dfAnos['provincia']== i ].ano, dfAnos[dfAnos['provincia']== i ].TotalPubl, color='blue')

        ax.plot(dfAnos[dfAnos['provincia']== i ].ano, dfAnos[dfAnos['provincia']== i ].nulls, color='red')



plt.xticks([2012,2013,2014,2015,2016])

plt.grid(b=True, which='major', axis='both')

ax.set_xlabel("\n Año", fontsize=14)

ax.set_ylabel("Cantidad \n", fontsize=14)

ax.legend(loc='best')    

ax.set_title('Publicaciones vs Nulls por año por provincia \n', fontdict={'fontsize':16})