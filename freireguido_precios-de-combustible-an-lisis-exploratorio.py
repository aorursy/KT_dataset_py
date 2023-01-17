import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



df = pd.read_csv('../input/precios-de-combustible-en-argentina/precios-en-surtidor-resolucin-3142016.csv')

df.indice_tiempo = pd.to_datetime(df.indice_tiempo)

df
#Creo un segundo dataframe con precios PROMEDIO de combustible por provincia a lo largo del tiempo

df2 = pd.DataFrame(columns=df.provincia.unique(), index=df.indice_tiempo.unique())

for tiempo, provincias in df2.iterrows():

    for provincia in provincias.keys():

        precios = df.loc[(df.indice_tiempo == tiempo) & (df.provincia == provincia) & (df.idproducto < 4 ), 'precio']

        if len(precios) != 0:

            df2.loc[tiempo, provincia] = sum(precios) / len(precios)

        else:

            df2.loc[tiempo, provincia] = np.nan

df2.sort_index(inplace=True)

df2
#Para mejorar la visualización utilizo un forward fill para rellenar los NaN

df2.fillna(method='ffill', inplace=True)

df2
#Ahora uso bar_chart_race para hacer el GIF

!pip install bar_chart_race

import bar_chart_race as bcr

bcr.bar_chart_race(df2.dropna(how='all').fillna(0), n_bars=5, steps_per_period=60, period_length=1000)