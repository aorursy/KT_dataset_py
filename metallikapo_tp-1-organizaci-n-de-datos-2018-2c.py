%matplotlib inline

import datetime as datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid')
sns.set(rc={'figure.figsize':(15.7,10.27)})
#Cargo los datos en memoria
eventos = pd.read_csv('../input/events.csv', low_memory=False)
eventos['timestamp'] =  pd.to_datetime(eventos['timestamp'])
eventos[['marca','modelo']] = eventos['model'].dropna().str.split(' ',n=1,expand=True)
eventos['weekday']=eventos['timestamp'].apply(lambda x: x.day_name())
eventos['hour']=eventos['timestamp'].dt.hour
eventos['month']=eventos['timestamp'].dt.month
eventos.head()
#Hay algun dato nulo?
eventos.isnull().any()
#hay alguna columna completamente nula?
eventos.isnull().all()
#Vamos a analizar algunas características de las columnas
eventos.describe()
#Existen 27624 usuarios únicos

cantidadDeUsuarios = eventos['person'].value_counts().count()
print('Cantidad de usuarios total: ', cantidadDeUsuarios)
#Existen 11 tipos de eventos distintos
eventos['event'].value_counts().count()
eventos['event'].value_counts()
#De los 33.735 de checkout, queremos saber si todos los usuarios únicos (27.624) hicieron al menos un checkout.

eventosCheckout = eventos.loc[eventos['event'] == 'checkout','person']
cantidadDeCheckout = eventosCheckout.count()
cantidadDeUsuariosCheckout = eventosCheckout.value_counts().count()
print(cantidadDeUsuariosCheckout,' realizaron al menos un checkout, de un total de ',cantidadDeUsuarios,' usuarios')
#Puede inferirse que los datos suministrados fueron de usuarios que al menos hicieron un checkout
#Viendo que todos hicieron checkout al menos una vez, vemos cuántos realizaron la compra. 
conversiones = eventos.loc[eventos['event'] == 'conversion','person']
cantidadDeConversiones = conversiones.count()
cantidadDePersonasQueConvierten = conversiones.value_counts().count()
print('Se realizaron',cantidadDeConversiones,'conversiones entre',cantidadDePersonasQueConvierten,'clientes')
#De los 27.624 usuarios, 716 hicieron compras
print('De',cantidadDeCheckout,'checkouts se realizaron',cantidadDeConversiones,'conversiones')
print('La tasa de conversion por checkout es de',(cantidadDeConversiones * 100 / cantidadDeCheckout),'%')
print('El porcentaje de usuarios que concreta una compra es de',\
      (cantidadDePersonasQueConvierten * 100 / cantidadDeUsuariosCheckout),'%')
notificacionesDeStock = eventos.loc[eventos['event'] == 'lead',['person','model']]
cantPersonasQueSeRegistraron = notificacionesDeStock['person'].value_counts().count()
print('Se registraron para recibir notificaciones',cantPersonasQueSeRegistraron,'personas')
#¿La gente se anotaba para recibir notificaciones de muchos modelos?
notificacionesDeStock['person'].value_counts().head(10)
#Modelos más requeridos que no estaban en stock
notificacionesDeStock['model'].value_counts().head(10)
eventos['search_term'].value_counts().head(15)
#Tratando de depurar un poco lás búsquedas
#eliminando diferencias por mayúsculas y minúsculas
eventos['search_term'] = eventos['search_term'].str.lower()
eventos['search_term'].value_counts().head(20)
#Se ve que lo que más se busca es iPhones. Del total de búsquedas, vamos a ver cuántas son de iPhones
totalDeBusquedas = eventos['search_term'].count()
busquedaIncluyeIphone = eventos['search_term'].str.contains('iphone').value_counts().to_frame('cantidad')
busquedaIncluyeIphone.rename({0:'falso',1:'verdadero'},inplace=True)
cantBusquedasIPhone = busquedaIncluyeIphone.loc['verdadero','cantidad']
print('Cantidad de busquedas total:',totalDeBusquedas)
print('Cantidad de búsquedas de iPhones:',cantBusquedasIPhone)
eventos['search_term'].str.lower().str.contains('iphone').value_counts(normalize=True)
sns.set(style="whitegrid")
plotBusquedaIphone = sns.barplot(x=busquedaIncluyeIphone['cantidad'],y=busquedaIncluyeIphone.index,orient='horizontal')
plotBusquedaIphone.set(title="Busquedas de Iphone",xlabel="Cant. de búsquedas",ylabel="Incluye palabra iphone")
#De las búsquedas que no son iPhone veamos cuales son las más frecuentes:
busquedaNoIncluyeIphone = eventos.loc[eventos['search_term'].str.contains('iphone') == False, 'search_term'].value_counts().to_frame('cantidad')
busquedaNoIncluyeIphone = busquedaNoIncluyeIphone.loc[busquedaNoIncluyeIphone['cantidad'] >= 50,'cantidad'].to_frame()
busquedaNoIncluyeIphone = busquedaNoIncluyeIphone.reset_index()
busquedaNoIncluyeIphone.rename(index=str, columns={'index': "busqueda"},inplace=True)
busquedaNoIncluyeIphone.head(30)
#Claves buscadas que no incluyen la palabra iPhone
plotBusquedaNoIphone = sns.barplot(x=busquedaNoIncluyeIphone['cantidad'],y=busquedaNoIncluyeIphone['busqueda'].head(20),orient='horizontal')
plotBusquedaNoIphone.set(title="Busquedas que no incluyen iPhone",xlabel="Cant. de búsquedas",ylabel="Clave buscada")
eventos.loc[:,['marca','modelo']].dropna().drop_duplicates().head(10)
modelos = eventos.loc[:,['model','marca']].dropna().drop_duplicates()
modelos['model'] = modelos['model'].str.lower()
modelos.head()
eventos['month'].value_counts().plot(kind='bar',figsize=(15,5),fontsize=12,rot=0)
plt.title("Trafico segun mes", size=20)
plt.xlabel("Mes",size=15)
plt.ylabel("Cantidad de eventos",size=15)
plt.show()
#Otra forma
cantidadEventosPorMes = eventos['month'].value_counts().to_frame('cantidad')
plotTraficoSegunMes = sns.barplot(x=cantidadEventosPorMes.index, y=cantidadEventosPorMes['cantidad'])
plotTraficoSegunMes.set(xlabel='Mes', ylabel='Cant. de eventos', title='Distribución de eventos por mes')
#Grafico Eventos segun dia de la semana
eventos.groupby('weekday')['event'].count().sort_values(ascending=False).plot(kind='bar',figsize=(15,5),fontsize=12,rot=0)
plt.title('Eventos segun dia de la semana')
plt.ylabel("Eventos", size=15)
plt.xlabel("Dia", size=15)
plt.show()
#Grafico Trafico Segun hora
eventos.groupby('hour')['hour'].count().plot(kind='line',figsize=(15,5),fontsize=12,rot=0)
plt.title('Trafico segun hora')
plt.ylabel("Eventos", size=15)
plt.xlabel("Hora", size=15)
plt.show()
#Distribución de eventos según hora
plotEventosSegunHora = sns.distplot(eventos['hour'])
plotEventosSegunHora.set(title='Distribución de eventos según hora',xlabel='Hora',ylabel='Distribución')
eventos.loc[(eventos['device_type']=='Computer')|(eventos['device_type']=='Smartphone')].groupby(['hour','device_type'])['hour'].count().unstack().plot(kind='bar',figsize=(15,5),fontsize=12,rot=0)
plt.title('Visitas segun horario, segun dispositivo')
plt.ylabel("Visitas", size=15)
plt.xlabel("Horario/Dispositivo", size=15)
plt.show()
eventos.loc[(eventos['event']=='brand listing')|(eventos['event']=='visited site')|(eventos['event']=='ad campaign hit')|(eventos['event']=='generic listing')|(eventos['event']=='searched products')|(eventos['event']=='search engine hit')|(eventos['event']=='checkout')].groupby(['hour','event'])['hour'].count().unstack().plot(kind='line',figsize=(20,10),fontsize=12)
plt.title('Visitas segun horario, segun dispositivo')
plt.ylabel("Visitas", size=15)
plt.xlabel("Horario/Dispositivo", size=15)
plt.show()

eventos.loc[eventos['event']=='viewed product']['storage'].value_counts().head(6).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Visualizaciones segun memoria')
plt.ylabel("Visualizaciones", size=15)
plt.xlabel("Memoria", size=15)
plt.show()
eventos.loc[eventos['event']=='conversion']['storage'].value_counts().head(4).plot(kind='bar',figsize=(15,15),fontsize=14,rot=0)
plt.title('Conversions segun memoria')
plt.ylabel("Conversions", size=15)
plt.xlabel("Memoria", size=15)
plt.show()

eventos.loc[eventos['event']=='viewed product']['color'].value_counts().head(9).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Visualizaciones segun color')
plt.ylabel("Visualizaciones", size=15)
plt.xlabel("Color", size=15)
plt.show()
eventos.loc[eventos['event']=='conversion']['color'].value_counts().head(9).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Conversions segun color')
plt.ylabel("Conversions", size=15)
plt.xlabel("Color", size=15)
plt.show()

eventos.loc[eventos['event']=='viewed product']['condition'].value_counts().head(9).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Visualizaciones segun condicion')
plt.ylabel("Visualizaciones", size=15)
plt.xlabel("Condicion", size=15)
plt.show()

eventos.loc[eventos['event']=='conversion']['condition'].value_counts().head(9).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Conversions segun condicion')
plt.ylabel("Conversions", size=15)
plt.xlabel("Condicion", size=15)
plt.show()

# Conversions segun Modelo
eventos.loc[eventos['event']=='viewed product'].groupby('model').count()['timestamp'].sort_values(ascending=False).head(10).plot(kind='bar',figsize=(15,5),fontsize=12)
plt.title('Visualizaciones segun modelo')
plt.ylabel("Visualizaciones", size=15)
plt.xlabel("Modelo", size=15)
plt.show()

# Conversions segun Modelo
eventos.loc[eventos['event']=='conversion'].groupby('model').count()['timestamp'].sort_values(ascending=False).head(10).plot(kind='bar',figsize=(15,5),fontsize=12)
plt.title('Conversions segun modelo')
plt.ylabel("Conversions", size=15)
plt.xlabel("Modelo", size=15)
plt.show()

#Grafico Numero Eventos segun tupla Evento/Marca
eventos.loc[eventos['event']=='viewed product'].groupby('marca')['timestamp'].count().sort_values(ascending=False).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Visualizaciones segun marca')
plt.ylabel("Visualizaciones", size=15)
plt.xlabel("Marca", size=15)
plt.show()
eventos.loc[eventos['event']=='conversion'].groupby('marca')['timestamp'].count().sort_values(ascending=False).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Conversions segun marca')
plt.ylabel("Conversions", size=15)
plt.xlabel("Marca", size=15)
plt.show()
#Grafico eventos segun region
eventos.loc[~(eventos['country']=='Unknown')].groupby('country')['event'].count().sort_values(ascending=False).head(5).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Trafico segun pais')
plt.ylabel("Visitas", size=15)
plt.xlabel("Pais", size=15)
plt.show()
#Grafico eventos segun region
eventos.loc[~(eventos['region']=='Unknown')].groupby('region')['event'].count().sort_values(ascending=False).head(10).plot(kind='bar',figsize=(15,5),fontsize=14)
plt.title('Trafico segun region')
plt.ylabel("Visitas", size=15)
plt.xlabel("Region", size=15)
plt.show()
#Grafico eventos segun ciudad
eventos.loc[~(eventos['city']=='Unknown')].groupby('city')['event'].count().sort_values(ascending=False).head(10).plot(kind='bar',figsize=(15,5),fontsize=11,rot=0)
plt.title('Trafico segun ciudad')
plt.ylabel("Visitas", size=15)
plt.xlabel("Ciudad", size=15)
plt.show()

eventos.loc[eventos['event']=='ad campaign hit'].groupby('campaign_source').count().head(5)['timestamp'].sort_values(ascending=False).plot(kind='bar',figsize=(15,5),fontsize=14,rot=0)
plt.title('Visitas segun campana')
plt.ylabel("Visitas", size=15)
plt.xlabel("Campana", size=15)
plt.show()
eventos.loc[eventos['event']=='ad campaign hit']['url'].value_counts().head(10).plot(kind='bar',figsize=(15,5),fontsize=12)
plt.title('Url accedidas mediante campana')
plt.ylabel("Visitas", size=15)
plt.xlabel("Url", size=15)
plt.show()

