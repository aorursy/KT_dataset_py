import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
def cambiarLabel(x):

    if (x == 2.0):

        return 'centro y escuela cerca'

    if (x == 1.0):

        return 'centro o escuela cerca'

    if (x == 0.0):

        return 'ni centro ni escuela cerca'

    else:

        return 'nan'
df['centroscomercialescercanos'].dropna(inplace = True)

df['escuelascercanas'].dropna(inplace = True)
df['categoria'] = df['centroscomercialescercanos'] + df['escuelascercanas']

df['categoria'] = df['categoria'].map(lambda x : cambiarLabel(x))
d = df.groupby('categoria').size()

d.head()
#"barplot con todas las combinaciones de si tiene o no escuela cercana y centro comercial cercano vs cantidad de inmuebles"



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_16 = sns.barplot(x = d.index, y = d.values, palette = 'coolwarm')



plt.title('Relacion entre centros comerciales cercanos y escuelas cercanas')

plt.xlabel('Combinacion booleana entre escuelas cercanas y centro comercial cercano')

plt.ylabel('cantidad de propiedades')