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
propiedades = getNormalizedDataset()
dfChangeRate = pd.read_csv('../input/mexican-zonaprop-datasets/usd_mxc.csv', parse_dates=['Fecha'], dtype={'Último': float}, decimal=",")

dfChangeRate = dfChangeRate.drop(['Apertura', 'Máximo', 'Mínimo', '% var.'], axis=1)
#Calculo el precio promedio del metro cuadrado por año

metroCuadradoPorAño = propiedades.copy()[['fecha', 'precio_m2']]

metroCuadradoPorAño.loc[:, 'año'] = metroCuadradoPorAño['fecha'].dt.year

metroCuadradoPorAño = metroCuadradoPorAño.groupby(['año']).agg({'precio_m2':'mean'})

metroCuadradoPorAño = metroCuadradoPorAño.sort_values(by=['año'])

metroCuadradoPorAño = metroCuadradoPorAño.reset_index()
#Calculo el precio promedio del dolar por año

dfChangeRate.loc[:, 'año'] = dfChangeRate['Fecha'].dt.year

dfChangeRate = dfChangeRate.groupby(['año']).agg({'Último':'mean'})

dfChangeRate = dfChangeRate.sort_values(by=['año'])

dfChangeRate = dfChangeRate.reset_index()
#Show 

fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(

    go.Scatter(x=metroCuadradoPorAño['año'], y=metroCuadradoPorAño['precio_m2'], name="Precio promedio metro cuadrado"),

    secondary_y=False,

)

fig.add_trace(

    go.Scatter(x=dfChangeRate['año'], y=dfChangeRate['Último'], name="Valor del dolar"),

    secondary_y=True,

)

fig.update_layout(

    title_text="Precio promedio por metro cuadrado vs Valor del dolar"

)

fig.update_xaxes(title_text="Años")



fig.update_yaxes(title_text="Precio promedio metro cuadrado", secondary_y=False)

fig.update_yaxes(title_text="Valor del dolar", secondary_y=True)



fig.show()