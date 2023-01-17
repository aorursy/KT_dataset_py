import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
def crearDiccionarioMapeandoCiudadConSuCategoria(df):

    

    dic = {}

    for ciudad in df['ciudad']:

        

        if not (ciudad in dic):

            n = np.log10((df['ciudad'] == ciudad).sum() + 1).astype(int)

            dic[ciudad] = n

        

    return dic
dic = crearDiccionarioMapeandoCiudadConSuCategoria(df)
df['categoria de ciudad'] = df['ciudad'].map(crearDiccionarioMapeandoCiudadConSuCategoria(df))
#Filtramos los outliners



df = df[df['precio'] <= 6000000]
#violinplot de grupos de ciudades de igual magnitud en props (log) vs precio



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_16 = sns.violinplot(data = df, x = 'categoria de ciudad', y = 'precio', palette = 'coolwarm', showfliers = False)



plt.title('Precio de ciudades con un mismo orden de magnitud')

plt.xlabel('orden de magnitud de cantidad de propiedades')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')