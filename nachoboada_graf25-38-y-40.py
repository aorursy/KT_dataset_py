import pandas as pd

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
data = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
df = pd.DataFrame(data)
df.columns.values
data0 = {'provincia':data['provincia']}
df0 = pd.DataFrame(data0)
df0 = df0.groupby(['provincia'])
df0.count()
df0.size()
data1 = {'tipodepropiedad':data['tipodepropiedad'],'precio':data['precio']}
df1 = pd.DataFrame(data1)
df1.head()
df1 = df1.groupby(['tipodepropiedad']).mean()

df1
df1 = df1.sort_values(['precio'], ascending=[False])
df1
df1.plot.bar()
df2 = pd.DataFrame(data1)
df2 = df2.groupby(['tipodepropiedad']).std()
df2 = df2.sort_values(['precio'], ascending=[False])
df2['precio'] = df2['precio'].map(lambda x: x/df2['precio'].max())
df2.plot.bar()
df3=data.pivot(index = 'id',columns = 'provincia',values = 'precio')
d1 = df.groupby('provincia')

serie = np.round(np.log(d1.size()+1))

d2 = pd.DataFrame(serie)

d2.columns.values

d3 = d2.groupby(0)

d3.head(20)
d3.get_group(7.0)
df3.head(20)
boxplot = df3.boxplot()
data2 = {'ciudad':data['ciudad']}

df2 = pd.DataFrame(data2)
df2.head()
df2['quantity'] = 1
df2 = df2.groupby('ciudad').sum()
df2
df2 = df2.sort_values(['quantity'], ascending=[False])

df2
df2 = df2.head(20)
df2.plot.bar()
plt.scatter(data.lat,data.lng, alpha='0.1') #Grafico la posicion de la pantalla donde clickean en los dispositivos

plt.title("Latitud vs Longitud", fontdict=dict(weight='bold'))

plt.xlabel("Latitud", fontdict=dict(weight='bold'))

plt.ylabel("Longitud", fontdict=dict(weight='bold'))

plt.show()
data2 = {'idzona':data["idzona"]} 
df2 = pd.DataFrame(data2)
df2.head()
df2['quantity'] = 1
df2.head()
df2 = df2.groupby(['idzona']).agg({'quantity':sum}) #Agrupo y sumo
df2 = df2.sort_values(['quantity'], ascending=[False]) #Ordeno a los operadores de más a menos relevante
df2.head()
df2 = df2.head(20)
df2
df2.plot.bar()