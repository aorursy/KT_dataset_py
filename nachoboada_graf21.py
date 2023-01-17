import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
#boxplot de tipo de propiedad vs metros cuadrados ocupados



plt.figure(figsize = (20, 5))

sns.set(font_scale = 2)



g_21 = sns.boxplot(x = 'tipodepropiedad', y = 'metroscubiertos', data = df)



g_21.set_xticklabels(g_21.get_xticklabels(), rotation = 90)



g_21



plt.title('Distribucion por tipo de propiedad de metros cuadrados cubiertos')

plt.xlabel('Tipo de propiedad')

plt.ylabel('Metros cuadrados cubiertos')