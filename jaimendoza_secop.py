# Estas líneas de comando importan las librerías y dependencias necesarias para el análisis.

import numpy as np

import pandas as pd

import plotly

plotly.__version__

import os

import plotly.plotly as py

import plotly.graph_objs as go

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
# En esta líne de código se importa el conjunto de datos a analizar.

secop = pd.read_csv('../input/SECOP_II_-_Contratos.csv', encoding='utf8')

secop.head()
# Dimensiones del Dataset

secop.shape
# Selección de las variables de interés

df = secop[["Nombre Entidad", "Tipo de Contrato", 

    "Fecha de Inicio del Contrato", "Fecha de Fin del Contrato", 

    "Fecha de Inicio de Ejecucion", "Fecha de Fin de Ejecucion", 

    "Proveedor Adjudicado", "Valor del Contrato",

   "Tipo de Proceso", "Es Post Conflicto",

    "Estado Contrato"]]

df
# Valor del contrato en billones de pesos

df["Valor del Contrato"] = pd.to_numeric(df["Valor del Contrato"])/1000000

df
# Cantidad de contratos por estado del mismo

c_contratos = df["Estado Contrato"].value_counts().to_frame()

print(c_contratos)

plt.figure(figsize=(10,10))

sns.barplot(y=c_contratos.index,x=c_contratos["Estado Contrato"])
t_contratos = df["Tipo de Contrato"].value_counts().to_frame()

print(t_contratos)

plt.figure(figsize=(10,10))

sns.barplot(y=t_contratos.index,x=t_contratos["Tipo de Contrato"])
# Valor de los contratos según el tipo 

v_contratos = df.groupby(["Tipo de Contrato"]).agg('sum')["Valor del Contrato"].to_frame().reset_index().sort_values(['Valor del Contrato'],ascending=False)

print(v_contratos)

plt.figure(figsize=(10,10))

sns.barplot(y=v_contratos["Tipo de Contrato"],x=v_contratos["Valor del Contrato"])
# Se aplican los filtros

df2 = df.loc[(df["Estado Contrato"] == "Activo") & (df["Tipo de Contrato"] != "Prestación de servicios")]

df2.shape
# Variables seleccionadas

df.columns
print(pd.unique(df2["Tipo de Contrato"]))

print(pd.unique(df2["Tipo de Proceso"]))

print(pd.unique(df2["Es Post Conflicto"]))
# Reemplazar valores nulos

df2["Es Post Conflicto"] = df2["Es Post Conflicto"].fillna(0)

print(pd.unique(df2["Es Post Conflicto"]))
# Cantidad de contratos por estado del mismo

pc_contratos = df2["Es Post Conflicto"].value_counts().to_frame()

print(pc_contratos)

plt.figure(figsize=(10,10))

sns.barplot(x=pc_contratos.index,y=pc_contratos["Es Post Conflicto"])
# Se aplica filtro de post conflicto

df3 = df2.loc[(df["Es Post Conflicto"] == 1.00)]

df3.shape
df3
tc_pc_contratos = df3["Tipo de Contrato"].value_counts().to_frame()

print(tc_pc_contratos)

plt.figure(figsize=(10,10))

sns.barplot(x=tc_pc_contratos.index,y=tc_pc_contratos["Tipo de Contrato"])
df2["Fecha de Inicio del Contrato"] = pd.to_datetime(df2["Fecha de Inicio del Contrato"])

df2["Fecha de Fin del Contrato"] = pd.to_datetime(df2["Fecha de Fin del Contrato"])
df2["año_fecha_inicio"] = df2["Fecha de Inicio del Contrato"].map(lambda x: x.year)

df2["mes_fecha_inicio"] = df2["Fecha de Inicio del Contrato"].map(lambda x: x.month)

df2["año_fecha_fin"] = df2["Fecha de Fin del Contrato"].map(lambda x: x.year)

df2["mes_fecha_fin"] = df2["Fecha de Fin del Contrato"].map(lambda x: x.month)
df2
r = df2["año_fecha_inicio"].value_counts().to_frame()

r = r.sort_index()

t = r.index.tolist()

s = r["año_fecha_inicio"].tolist()

print(t)

print(s)

fig, ax = plt.subplots()

ax.plot(t, s)



ax.set(xlabel='Tiempo (Años)', ylabel='Cantidad Contratos')

ax.grid()



plt.show()
r = df2["año_fecha_fin"].value_counts().to_frame()

r = r.sort_index()

t = r.index.tolist()

s = r["año_fecha_fin"].tolist()

print(t)

print(s)

fig, ax = plt.subplots()

ax.plot(t, s)



ax.set(xlabel='Tiempo (Años)', ylabel='Cantidad Contratos')

ax.grid()



plt.show()