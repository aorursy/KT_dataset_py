import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
#Cantidad de propiedades por ciudad



d = df.groupby('ciudad').size()

d.sort_values(ascending = False).head(20)
#Orden de magnitud en propiedades por ciudad



d = np.log10(d + 1).astype(int)

d.sort_values(ascending = False).head(20)
#Me hago un data frame con los datos



d = pd.DataFrame({'ciudades':d.index, 'orden de magnitud en propiedades por ciudad':d.values})

d.sort_values(by = 'orden de magnitud en propiedades por ciudad',ascending = False).head(20)
#Agrupo en grupos con igual orden de magnitud



d = d.groupby('orden de magnitud en propiedades por ciudad').size()

d.head(20)
#"magnitud de props(log) vs cant de ciudades con magnitud de props(log)"



plt.figure(figsize = (40, 20))

sns.set(font_scale = 4)



g_16 = sns.barplot(x = d.index, y = d.values, palette = 'coolwarm')



plt.title('Cantidad de ciudades con un mismo orden de magnitud')

plt.xlabel('orden de magnitud de cantidad de propiedades')

plt.ylabel('cantidad de ciudades')