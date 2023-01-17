#El origen de los datos es un dataset creado a partir de datos de la pagina HostelWorld
#Objetivo del análisis: Quiero viajar a Japon el proximo verano, de mochilero:
#Me interesa un alojamiento barato y limpio.
#Como usaré transporte público ha de estar centrico.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# hostales en Japon
df_hostales = pd.read_table('../input/Hostel.csv', sep=',', header=0)
df_hostales.head(5)
# ¿realmente es recomendable dormir en hostales en Japon?
#grafico de puntuaciones de hostales segun listado
df_hostales['summary_score'].value_counts().sort_index().plot.line()

#Obtencion de cuantos hostales tienen una calidad/precio mayor al 90%
df_hostales_smart = df_hostales[(df_hostales.summary_score > 9)]
df_hostales_smart_agg = df_hostales_smart.groupby(['City'])['City'].agg(['count'])
df_hostales_smart_agg
#Obtencion de los hostales más centricos en Osaka

df_osaka_centro = df_hostales_smart[(df_hostales_smart.City == 'Osaka') &
                             (df_hostales_smart.Distance < 2)]
df_osaka_centro.head(3)
# Obtencion del precio en euros de los hostales en Hiroshima, ordenados por precio

df_hostales['eur_price'] = df_hostales.apply(lambda row: row.price / 130, axis=1)
df_hostales_hiroshima = df_hostales[df_hostales.City == 'Hiroshima']
df_hostales_hiroshima.head()
# Obtencion de los hostales en Tokio, añadiendo una columna sobre la limpieza.
# Obtencion de tres hostales en Tokio centricos, baratos y Muy limpios.

df_hostales_Tokio = df_hostales[df_hostales.City == 'Tokyo']

df_hostales_Tokio['Limpios'] = df_hostales_Tokio['cleanliness'] \
                        .apply(lambda x: 'Sucio' if x < 7 else ('MUY Limpio' if x > 9 else 'Aceptable'))
df_hostales_Tokio_cheap=df_hostales_Tokio[(df_hostales_Tokio.Limpios == 'MUY Limpio') &
                             (df_hostales_Tokio.Distance < 2) &
                                        (df_hostales_Tokio.eur_price < 20)]

df_hostales_Tokio_cheap.head()

#Obtencion del precio medio de los hostales en japon

df_precio_medio=media = df_hostales["eur_price"].median()

print("El precio medio es: %d €" % (df_precio_medio))
# Grafico de los mejores hostales (rating>90%) por ciudad 
df_hostales_smart['City'].value_counts().head(10).plot.bar()
y_city = df_hostales['City'].values
X_price = df_hostales['price'].values
X_price = np.reshape(np.asarray(X_price), (len(X_price), 1))
fig, ax =  plt.subplots(figsize=(12, 8))
ax.scatter(X_price, y_city,  color='red', marker='.', )
ax.set_title("Precio en funcion de la ubicacion")
ax.set_xlabel("precio")
ax.set_ylabel("ciudad")
plt.show()
