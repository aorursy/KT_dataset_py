# importacion general de librerias y de visualizacion (matplotlib y seaborn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

%matplotlib inline

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn

pd.options.display.float_format = '{:20,.2f}'.format # suprimimos la notacion cientifica en los outputs

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/events.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], errors = 'coerce')
eventosConPais = df[['event', 'country', 'region', 'city']].dropna()
eventosConPais.head()
eventosConPais['event'].value_counts()

#El unico evento que guarda datos sobre la ubicacion de usuarios es una visita al sitio
eventosConPais['country'].value_counts()

#La empresa es ampliamente mas grande en Brasil que en cualquier otro pais
#Elimino los eventos con ciudad desconocida

eventosFiltradosPorCiudad = eventosConPais.loc[eventosConPais['city'] != 'Unknown', :]

plotCiudades = eventosFiltradosPorCiudad['city'].value_counts().head(7).plot(kind = 'bar', figsize=(7,4))

plotCiudades.set_title("Ciudades con más visitas a la pagina", fontsize=18)
plotCiudades.set_xlabel("Ciudad",fontsize=18)
plotCiudades.set_ylabel("Cantidad de visitas", fontsize=18)

#La ciudad con más trafico en la pagina es Sao Paulo por un margen grande
productos = df[['event', 'model']].dropna()
productos.head()
eventosDeProductos = productos['event'].value_counts()

plotEventosDeProductos = np.log(eventosDeProductos).plot(kind = 'bar')

plotEventosDeProductos.set_title("Comparación entre vista de productos y compras", fontsize=18)
plotEventosDeProductos.set_ylabel("Cantidad de eventos (log)",fontsize=18)
plotEventosDeProductos.set_xlabel("Tipo de evento",fontsize=18)
productosVistos = productos.loc[productos['event'] == 'viewed product', :]

plotProductosVistos = productosVistos['model'].value_counts().head(10).plot(kind = 'bar', figsize=(10,4))

plotProductosVistos.set_title('Visitas de los 10 productos más populares', fontsize=18)
plotProductosVistos.set_ylabel('Vistas', fontsize=18)
plotProductosVistos.set_xlabel('Modelo', fontsize=18)
productosVendidos = productos.loc[productos['event'] == 'conversion', :]

plotProductosVendidos = productosVendidos['model'].value_counts().head(10).plot(kind = 'bar', figsize=(10,4))

plotProductosVendidos.set_title('Ventas de los 10 productos más vendidos', fontsize=18)
plotProductosVendidos.set_ylabel('Ventas', fontsize=18)
plotProductosVendidos.set_xlabel('Modelo', fontsize=18)
comparacionEventosDeProductos = sns.countplot(x = "model", hue = "event", data = productos, palette = "hls", order = productos.model.value_counts().iloc[:4].index)

comparacionEventosDeProductos.set_title('Comparacion entre visita a producto y compra de productos más populares', fontsize=18)
comparacionEventosDeProductos.set_xlabel('Producto', fontsize=18)
comparacionEventosDeProductos.set_ylabel('Cantidad de eventos (log)', fontsize=18)
comparacionEventosDeProductos.set_yscale('log')
ventas = df[['timestamp', 'event']]

ventas = ventas.loc[ventas['event'] == 'conversion', :]

ventas['timestamp'] = ventas['timestamp'].dt.date

plotVentas = ventas.groupby('timestamp').count()['event'].plot(figsize=(10,4))

plotVentas.set_title('Ventas en funcion del tiempo', fontsize=18)
plotVentas.set_ylabel('Ventas por día', fontsize=18)
plotVentas.set_xlabel('Fecha', fontsize=18)
campaniasPublicitarias = df['campaign_source'].dropna()

plotPublicidad = campaniasPublicitarias.value_counts().head(5).plot(kind = 'bar')

plotPublicidad.set_title('Cantidad de hits de las 5 mejores campañas publicitarias', fontsize=18)
plotPublicidad.set_xlabel('Campaña', fontsize=18)
plotPublicidad.set_ylabel('Hits', fontsize=18)

#Google es la que trae más hits por mucho
campaniasConTiempo = df[['timestamp', 'campaign_source']].dropna()

campaniasConTiempo['timestamp'] = campaniasConTiempo['timestamp'].dt.date

plotCampanias = campaniasConTiempo.groupby('timestamp').count()['campaign_source'].plot(figsize=(10,4))

plotCampanias.set_title('Cantidad de hits de campañas publicitarias', fontsize=18)
plotCampanias.set_xlabel('Tiempo', fontsize=18)
plotCampanias.set_ylabel('Hits', fontsize=18)
