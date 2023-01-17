# Importamos librerías de análisis de datos

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as datetime

import sys



import plotly.express as px

import plotly.graph_objects as go

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.figure_factory as ff



from normalize_data import getNormalizedDataset
#Cargamos los dataframe

propiedades = getNormalizedDataset()
#Creo dataframe que tenga como columnas tipoDePropiedad y los amenities

amenitiesPorPropiedad = propiedades[['tipodepropiedad','piscina','gimnasio', 'usosmultiples']]



#Calculo la media de cada amenitie por propiedad

amenitiesPorPropiedad = amenitiesPorPropiedad.groupby(['tipodepropiedad']).mean()

amenitiesPorPropiedad.dropna(inplace=True)



#Elimino los registros que tienen cero en algun amenitie, si no están en el grafico no tienen amenitie

#TODO: Improve filter method

#amenitiesPorPropiedad = amenitiesPorPropiedad.loc[lambda x: x['gimnasio'] != 0.000000, :].loc[lambda x: x['usosmultiples'] != 0.000000, :].loc[lambda x: x['piscina'] != 0.000000, :]

amenitiesPorPropiedad = amenitiesPorPropiedad[amenitiesPorPropiedad['gimnasio'] != 0.0]

amenitiesPorPropiedad = amenitiesPorPropiedad[amenitiesPorPropiedad['usosmultiples'] != 0.0]

amenitiesPorPropiedad = amenitiesPorPropiedad[amenitiesPorPropiedad['piscina'] != 0.0]

amenitiesPorPropiedad = amenitiesPorPropiedad.T * 100
print("Piscina por tipo de propiedad")

fig = go.Figure()

df = pd.DataFrame(dict(

    r=amenitiesPorPropiedad.loc['piscina'],

    theta=amenitiesPorPropiedad.columns))

fig = px.line_polar(df, r='r', theta='theta', line_close=True)

fig.update_traces(fill='toself')

fig.show()
print("Gimnasio por tipo de propiedad")

fig = go.Figure()

df = pd.DataFrame(dict(

    r=amenitiesPorPropiedad.loc['gimnasio'],

    theta=amenitiesPorPropiedad.columns))

fig = px.line_polar(df, r='r', theta='theta', line_close=True)

fig.update_traces(fill='toself')

fig.show()
print("Usosmultiples por tipo de propiedad")

fig = go.Figure()

df = pd.DataFrame(dict(

    r=amenitiesPorPropiedad.loc['usosmultiples'],

    theta=amenitiesPorPropiedad.columns))

fig = px.line_polar(df, r='r', theta='theta', line_close=True)

fig.update_traces(fill='toself')

fig.show()
def add_trace(fig, key):

    fig.add_trace(go.Scatterpolar(

      r=amenitiesPorPropiedad.loc[key],

      theta=categories,

      fill='none',

      name=key.title()

    ))
categories = amenitiesPorPropiedad.columns



fig = go.Figure()





add_trace(fig, 'piscina')

add_trace(fig, 'gimnasio')

add_trace(fig, 'usosmultiples')



fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 30]

    )),

  showlegend=True

)



fig.show()
amenitiesPorProvincia = propiedades[['provincia','piscina','gimnasio','usosmultiples']]

amenitiesPorProvincia = amenitiesPorProvincia.groupby(['provincia']).mean() * 100
plt.rcParams.update({'font.size': 18})

amenitiesPorProvincia.plot(kind='bar', figsize=(20, 10), rot=90, title="Amenities mas comunes en cada provincia");
#Relacion amenities por cantidad de habitaciones

amenitiesPorHabitaciones = propiedades[['habitaciones','piscina','gimnasio','usosmultiples']]

amenitiesPorHabitaciones = amenitiesPorHabitaciones.rename(columns={"habitaciones": "Propiedades(Por cantidad de habitaciones)"})

amenitiesPorHabitaciones = amenitiesPorHabitaciones.groupby(['Propiedades(Por cantidad de habitaciones)']).sum()

amenitiesPorHabitaciones.index = pd.to_numeric(amenitiesPorHabitaciones.index, downcast='integer')
amenitiesPorHabitaciones.plot(kind='bar', figsize=(15, 10), rot=0, 

                                       title="Cantidad de propiedades con Piscina/Gimnasio/Usosmultiples");
#Dentro de los departamentos de 3 ambientes que son los que mas amenities tienen, como varia el precio en relacion a los amenities? 

#Osea, si tiene 1 amenitie solo sale tanto, si tiene 2 sale tanto y si tiene 3 ?



amenitiesPorHabitaciones = propiedades[['habitaciones','piscina','gimnasio','usosmultiples', 'precio']]

amenitiesPorHabitaciones = amenitiesPorHabitaciones.rename(columns={"habitaciones": "Propiedades(Por cantidad de habitaciones)"})



propsNingunAmenitie = amenitiesPorHabitaciones.loc[lambda x: ~x['piscina'] & ~x['gimnasio'] & ~x['usosmultiples']]

propsNingunAmenitie = propsNingunAmenitie.groupby(['Propiedades(Por cantidad de habitaciones)']).mean()['precio'].to_frame()



propsUnAmenitie = amenitiesPorHabitaciones.loc[lambda x: x['piscina'] & ~x['gimnasio'] & ~x['usosmultiples'] \

                             | ~x['piscina'] & x['gimnasio'] & ~x['usosmultiples']

                             | ~x['piscina'] & ~x['gimnasio'] & x['usosmultiples'], :]

propsUnAmenitie = propsUnAmenitie.groupby(['Propiedades(Por cantidad de habitaciones)']).mean()['precio'].to_frame()



propsDosAmenitie = amenitiesPorHabitaciones.loc[lambda x: x['piscina'] & x['gimnasio'] & ~x['usosmultiples'] \

                             | x['piscina'] & ~x['gimnasio'] & x['usosmultiples']

                             | ~x['piscina'] & x['gimnasio'] & x['usosmultiples'], :]

propsDosAmenitie = propsDosAmenitie.groupby(['Propiedades(Por cantidad de habitaciones)']).mean()['precio'].to_frame()



propsTresAmenitie = amenitiesPorHabitaciones.loc[lambda x: x['piscina'] & x['gimnasio'] & x['usosmultiples']]

propsTresAmenitie = propsTresAmenitie.groupby(['Propiedades(Por cantidad de habitaciones)']).mean()['precio'].to_frame()



joined = propsNingunAmenitie.join(propsUnAmenitie, on='Propiedades(Por cantidad de habitaciones)', lsuffix='_sin_amenities', rsuffix='_un_amenitie')

joined = joined.join(propsDosAmenitie, on='Propiedades(Por cantidad de habitaciones)', lsuffix='_sin_amenities', rsuffix='_un_amenitie')

joined = joined.join(propsTresAmenitie, on='Propiedades(Por cantidad de habitaciones)', lsuffix='_dos_amenities', rsuffix='_tres_amenities')



#Se completan los nans con el valor obtenido de calcular el precio de la columna anterior mas la diferencia de ese con su anterior a la vez

joined["precio_dos_amenities"].fillna(joined["precio_un_amenitie"] + (joined["precio_un_amenitie"] - joined["precio_sin_amenities"]), inplace = True) 

joined["precio_tres_amenities"].fillna(joined["precio_dos_amenities"] + (joined["precio_dos_amenities"] - joined["precio_un_amenitie"]), inplace = True) 

joined = joined.rename(columns={"precio_sin_amenities": "Precio sin amenities", \

                                "precio_un_amenitie": "Precio con un amenitie", \

                                "precio_dos_amenities": "Precio con dos amenities", \

                                "precio_tres_amenities": "Precio con tres amenities"})

joined.index = pd.to_numeric(joined.index, downcast='integer')
plt.rcParams.update({'font.size': 18})

joined.plot(kind='bar', figsize=(15, 10), rot=0, 

                                       title="Crecimiento del valor promedio de la propiedad en función de la cantidad de amenities");
amenitiesPorAño = propiedades[['piscina', 'gimnasio', 'usosmultiples', 'fecha']]

amenitiesPorAño['año'] = amenitiesPorAño['fecha'].dt.year

amenitiesPorAño = amenitiesPorAño.groupby(['año']).sum()
fig, ax1 = plt.subplots(1,1, figsize=(15,5))



fig.subplots_adjust(hspace=0.4, wspace=0.4)

fig.suptitle('Crecimiento de amenities por año',fontsize=20)



amenitiesPorAño = amenitiesPorAño.reset_index()



for i in ['piscina', 'gimnasio', 'usosmultiples']: 

    ax1.plot(amenitiesPorAño.año,\

            amenitiesPorAño[i],\

            label=i)



plt.xticks([2012,2013,2014,2015,2016])    

ax1.set_xlabel("\n Año", fontsize=18)

ax1.set_ylabel("Cantidad de amenities\n", fontsize=18)

ax1.legend(loc='best', title_fontsize=16, fontsize=14)