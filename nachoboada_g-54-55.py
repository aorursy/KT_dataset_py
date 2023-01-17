import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
df.replace([np.inf, -np.inf], np.nan)

df['antiguedad'].dropna(inplace = True)
df['antiguedad'] = (df['antiguedad']).astype(int,errors = 'ignore')
d = df.groupby('antiguedad').size()
d = d.to_frame()

d.columns = ['cantidad']

d['antiguedad'] = d.index

d = d.sort_values(by = 'cantidad', ascending = False).head(40)
#grafico de barras antiguedad vs precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 2)



g_54 = sns.barplot(data = df, x = 'antiguedad', y = 'precio')



g_54.set(ylim=(0))

g_54.set_xticklabels(g_54.get_xticklabels(), rotation = 60)



plt.title('Relacion entre el precio y la antiguedad de las propiedades')

plt.xlabel('Antiguedad (Años)')

plt.ylabel('Precio promedio')



plt.ticklabel_format(style='plain', axis='y')
#grafico de barras antiguedad vs cantidad de propiedades con esa antiguedad



plt.figure(figsize = (40, 20))

sns.set(font_scale = 2)



g_55 = sns.barplot(data = d, x = 'antiguedad', y = 'cantidad')



g_55.set(ylim=(0))

g_55.set_xticklabels(g_55.get_xticklabels(), rotation = 60)



plt.title('Relacion entre el cantidad de propiedades y la antiguedad de las propiedades')

plt.xlabel('Antiguedad (Años)')

plt.ylabel('cantidad de propiedades con esa antiguedad')



plt.ticklabel_format(style='plain', axis='y')