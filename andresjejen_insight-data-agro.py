import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid") # estilo de salida de las gráficas

from datetime import datetime



import os

print(os.listdir("../input"))
Datos = pd.read_csv('../input/DatosAgro.txt',delimiter='\t')

Datos = Datos.drop(['Unnamed: 6'],axis=1)

Datos.head()
Datos['Fecha'] = Datos.Fecha.map(lambda x : datetime.strptime(x, '%d/%m/%Y'))



# reindexar el dataframe

Datos.index = Datos.Fecha

Datos.head()
for j in range(5):

    producto = Datos.Nombre_producto.drop_duplicates().values.tolist()[j]

    print(producto)



    Datos_resample_day = Datos[Datos.Nombre_producto == producto].resample('d').count()



    df_resample = pd.concat([Datos_resample_day], axis=1)

    df_resample['dayofweek'] = df_resample.index.dayofweek # 0 es lunes



    for i in range(44):

        df_resample.Producto[i*7:i*7+8].plot(figsize=(30,10),title="Historial de "+ producto)

    plt.show()

    df_resample.Producto.describe()