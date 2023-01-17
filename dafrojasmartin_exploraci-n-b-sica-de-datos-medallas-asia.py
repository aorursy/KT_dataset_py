import pandas as pd

import numpy as np

from decimal import Decimal

import matplotlib.pyplot as plt

import seaborn as sn

import os

import plotly.graph_objs as go

import plotly.offline as py

DataSet = pd.read_csv('../input/asiangamestop10.csv')
colnum = len(DataSet)

print("colnum :", TotalRowCount)

DataSet.head(10)

print(DataSet.head())

print(DataSet.info())

print(DataSet.shape)
DataSet.dtypes
DataSet.describe()

DataSet.rename(columns={'Year' : 'Año',}, inplace=True)

DataSet.rename(columns={'NOC' : 'Pais',}, inplace=True)

DataSet.rename(columns={'Gold' : 'Oros',}, inplace=True)

DataSet.rename(columns={'Silver' : 'Platas',}, inplace=True)

DataSet.rename(columns={'Bronze' : 'Bronces',}, inplace=True)

DataSet.head(10)
import seaborn as sns 

fig = plt.figure(figsize=(20,3))

sns.heatmap(DataSet.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
Numeropaises = DataSet['Pais'].unique()

print("Paises participantes")

print(Numeropaises)

a = len(Numeropaises)

print (a)
TotalPremios = DataSet.groupby(['Año'])['Total'].sum().nlargest(10)

print("Los 10 años con más medallas")

print(TotalPremios)

plt.figure(figsize=(22,7))

GraphData=DataSet.groupby(['Año'])['Total'].sum().nlargest(10)

GraphData.plot(kind='bar')

plt.ylabel('Conteo de Medallas')

plt.xlabel('Año')
TotalPaises = DataSet.groupby(['Pais'])['Total'].sum().nlargest(10)

print("Los 10 paises con más medallas")

print(TotalPaises)

plt.figure(figsize=(22,7))

GraphData=DataSet.groupby(['Pais'])['Total'].sum().nlargest(10)

GraphData.plot(kind='bar')

plt.ylabel('Conteo de Medallas')

plt.xlabel('Pais')
TotalPaises = DataSet.groupby(['Pais'])['Oros'].sum().nlargest(10)

print("Los 10 paises con más Oros")

print(TotalPaises)

plt.figure(figsize=(22,7))

GraphData=DataSet.groupby(['Pais'])['Oros'].sum().nlargest(10)

GraphData.plot(kind='bar')

plt.ylabel('Conteo de Oros')

plt.xlabel('Pais')
TotalPaises = DataSet.groupby(['Pais'])['Platas'].sum().nlargest(10)

print("Los 10 paises con más Platas")

print(TotalPaises)

plt.figure(figsize=(22,7))

GraphData=DataSet.groupby(['Pais'])['Platas'].sum().nlargest(10)

GraphData.plot(kind='bar')

plt.ylabel('Conteo de Platas')

plt.xlabel('Pais')
TotalPaises = DataSet.groupby(['Pais'])['Bronces'].sum().nlargest(10)

print("Los 10 paises con más Bronces")

print(TotalPaises)

plt.figure(figsize=(22,7))

GraphData=DataSet.groupby(['Pais'])['Bronces'].sum().nlargest(10)

GraphData.plot(kind='bar')

plt.ylabel('Conteo de Bronces')

plt.xlabel('Pais')