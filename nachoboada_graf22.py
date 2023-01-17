import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
df['porcentaje de metros cuadrados cubiertos'] = df['metroscubiertos']/df['metrostotales']

#df['porcentaje de metros cuadrados cubiertos'] = df['porcentaje de metros cuadrados cubiertos'].map(lambda x : (x*100))
df = df[df['porcentaje de metros cuadrados cubiertos'] <= 1]
#boxplot de tipo de propiedad vs porcentaje de metros cuadrados ocupados



plt.figure(figsize = (20, 5))

sns.set(font_scale = 2)



g_22 = sns.boxplot(x = 'tipodepropiedad', y = 'porcentaje de metros cuadrados cubiertos', data = df, showfliers = False)



g_22.set_xticklabels(g_22.get_xticklabels(), rotation = 90)



g_22



plt.title('Distribucion por tipo de propiedad de porcentaje de metros cuadrados cubiertos')

plt.xlabel('Tipo de propiedad')

plt.ylabel('Porcentaje de metros cuadrados cubiertos')