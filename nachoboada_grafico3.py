import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline



trainPath = "/kaggle/input/trainparadatos/train.csv"
df = pd.read_csv(trainPath, index_col=0)
d = df.pivot(columns = 'tipodepropiedad', values= 'precio')

d.shape
#Intentamos limpiar la data y vemos que ya esta limpia



d.dropna(how = 'all')

d.shape
"tipo de prop en boxplot con precio"



plt.figure(figsize = (40, 20))

sns.set(font_scale = 3)



g_3 = sns.boxplot(x = df.tipodepropiedad, y = df.precio, palette = 'Set1', saturation = 80, showfliers = False)



g_3.set_xticklabels(g_3.get_xticklabels(), rotation = 90)



plt.title('Distribucion de precios por cada tipo de propiedad')

plt.xlabel('Tipo de propiedad')

plt.ylabel('Precio')

plt.ticklabel_format(style='plain', axis='y')



g_3
data = pd.read_csv(trainPath)

data20 = {'tipodepropiedad':data['tipodepropiedad'],'precio':data['precio']}
data.loc[data['precio'] <= 0]
df20 = pd.DataFrame(data20)
df20 = df20[df20["tipodepropiedad"].isin(['Apartamento','Casa','Casa en condominio','Casa uso de suelo','Departamento Compartido','Duplex'])]
plt.figure(figsize=(40,20))

sns.set(font_scale = 3)

g = sns.boxplot(x="tipodepropiedad", y="precio",data=df20,dodge=False)

plt.title('Variación del precio de las propiedades residenciales')

plt.xlabel('Propiedad')

plt.ylabel('Precio')







plt.ticklabel_format(style='plain', axis='y')
