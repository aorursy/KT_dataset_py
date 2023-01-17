import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0, parse_dates = ['fecha'])
#Se parsea las fechas a formato Timestamp



type(df['fecha'].iloc[0])
df['mes'] = df['fecha'].map(lambda x: x.month)

df['año'] = df['fecha'].map(lambda x: x.year)

df['cantidad de publicaciones'] = 1
df.sort_values(by = 'mes', ascending = True)
d = pd.pivot_table(df, values = 'cantidad de publicaciones', index = 'año', columns = 'mes', aggfunc = np.sum)
#heatmap con eje x dia de semana, eje y semanas, eje z cantidad de publicaciones



plt.figure(figsize = (20, 5))

sns.set(font_scale = 2)



plt.title('Cantidad de publicaciones por año y mes')

plt.xlabel('Mes')

plt.ylabel('Año')



g_20 = sns.heatmap(d, vmin = 0, cmap = 'rainbow_r')



g_20.set_yticklabels(g_20.get_yticklabels(), rotation = 0)