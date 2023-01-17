import numpy as np 
import pandas as pd

# plots
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/events.csv')
len(df)
df.sample(1)
df.dtypes
# Le agregamos 3 columnas, una de año, otra de dia de la semana y otra de hora.
import calendar
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Year'] = df['timestamp'].map(lambda x:x.year)
df['Weekday'] = df['timestamp'].map(lambda x:x.weekday_name)
df['Hour'] = pd.to_datetime(df['timestamp'], format='%H:%M',errors='coerce').dt.hour



cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df['Weekday'] = df['Weekday'].astype('category', categories=cats, ordered=True)
df.info()

# Numero de eventos por hora
plotEH = df.groupby('Hour').count()['event'].plot(kind='bar',figsize=(14,4));

plotEH.set_title("Numero de eventos por hora", fontsize=18)
plotEH.set_xlabel("Horas",fontsize=18)
plotEH.set_ylabel("Eventos", fontsize=18)

df.groupby('Hour').count()['event']
# Numero de eventos por dia

plotED=df.groupby('Weekday').count()['event'].plot(kind='bar',figsize=(14,4));

plotED.set_title("Numero de eventos por dia", fontsize=18)
plotED.set_xlabel("Dias",fontsize=18)
plotED.set_ylabel("Eventos", fontsize=18)
df.groupby('Weekday').count()['event']
#paises y eventos
dfFiltrado = df.loc[df['country'] != 'Unknown', :]

#plotC=df.groupby('country').count()['event'].plot(kind='bar',figsize=(14,4));
#Este hace algo parecido

plotC=dfFiltrado['country'].value_counts()[0:4].plot(kind='bar',figsize=(14,4));
plotC.set_title("Eventos en los paises", fontsize=18)
plotC.set_xlabel("Paises",fontsize=18)
plotC.set_ylabel("Eventos", fontsize=18)
plotC.set_yscale('log')


dfFiltrado['country'].value_counts()
#regiones y eventos
dfFiltrado = df.loc[df['region'] != 'Unknown', :]
plotR=dfFiltrado['region'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));
plotR.set_title("Eventos en las regiones", fontsize=18)
plotR.set_xlabel("Regiones",fontsize=18)
plotR.set_ylabel("Eventos", fontsize=18)
dfFiltrado['region'].value_counts()[0:30]
#ciudades y eventos
dfFiltrado = df.loc[df['city'] != 'Unknown', :]
plotF=dfFiltrado['city'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));

plotF.set_title("Numero de eventos por ciudad", fontsize=18)
plotF.set_xlabel("Ciudad",fontsize=18)
plotF.set_ylabel("Eventos", fontsize=18)
compras = df.loc[df.event.str.contains('conversion'),:]
compras
#dataframe de los eventos que son "conversion" (Compras)

plotH=compras.groupby('Hour').count()['event'].plot(kind='bar',figsize=(14,4));
#numero de compras por hora
plotH.set_title("Numero de compras por hora", fontsize=18)
plotH.set_xlabel("Hora",fontsize=18)
plotH.set_ylabel("Compra", fontsize=18)
compras.groupby('Hour').count()['event']
plotD=compras.groupby('Weekday').count()['event'].plot(kind='bar',figsize=(14,4));
plotD.set_title("Numero de compras por dia", fontsize=18)
plotD.set_xlabel("Dia",fontsize=18)
plotD.set_ylabel("Compras", fontsize=18)
#numero de compras por dia de la semana
compras.groupby('Weekday').count()['event']
#Analisis por año de los diferentes eventos

# tipo de eventos por año (2018)...

plot=pd.crosstab(df.Year, df.event).plot(kind='bar',figsize=(14,4));
plot.set_title("Tipos de evento en el 2018", fontsize=18)
plot.set_xlabel("Year",fontsize=18)
plot.set_ylabel("Cantidad de eventos", fontsize=18)
plot.set_yscale('log')



pd.crosstab(df.Year, df.event)

plot=pd.crosstab(df.Weekday, df.event).plot(kind='bar',figsize=(14,4));
plot.set_title("Comparacion de eventos en la semana", fontsize=18)
plot.set_xlabel("Weekday",fontsize=18)
plot.set_ylabel("Cantidad de eventos", fontsize=18)


pd.crosstab(df.Weekday, df.event)

plotHora=pd.crosstab(df.Hour, df.event).plot(kind='bar',figsize=(14,4));
plotHora.set_title("Comparacion de eventos a cada hora", fontsize=18)
plotHora.set_xlabel("Horas",fontsize=18)
plotHora.set_ylabel("Cantidad de eventos", fontsize=18)

pd.crosstab(df.Hour, df.event)
# tipos de dispositivos de donde se origino el evento

dfFiltrado = df.loc[df['device_type'] != 'NaN', :]
dfFiltrado = df.loc[df['device_type'] != 'Unknown', :]

plotCEL=dfFiltrado['device_type'].value_counts()[0:5].plot(kind='bar',figsize=(14,4));
plotCEL.set_title("Tipos de dispositivos de donde se origino el evento", fontsize=18)
plotCEL.set_xlabel("Tipos de dispositivos",fontsize=18)
plotCEL.set_ylabel("Cantidad de eventos", fontsize=18)

dfFiltrado['device_type'].value_counts()[0:5]
pd.crosstab(df.device_type,df.event)
# Lista de eventos, en este caso solo se lista visited site.
#tipos de resoluciones de los dispositivos donde se origino el evento
dfFiltradoR = df.loc[df['screen_resolution'] != 'NaN', :]
dfFiltradoR = df.loc[df['screen_resolution'] != 'Unknown', :]
plotCELR=dfFiltradoR['screen_resolution'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));
plotCELR.set_title("Tipos de resolucion de los dispositivos donde se origino el evento", fontsize=18)
plotCELR.set_xlabel("Resolucion",fontsize=18)
plotCELR.set_ylabel("Cantidad de eventos", fontsize=18)

dfFiltradoR['screen_resolution'].value_counts()[0:5]
pd.crosstab(df.screen_resolution,df.event)
# tipos de so de los dispositivos donde se origino el evento. 
dfFiltradoSO = df.loc[df['operating_system_version'] != 'NaN', :]
dfFiltradoSO = df.loc[df['operating_system_version'] != 'Unknown', :]
plotCELSO=dfFiltradoSO['operating_system_version'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));
plotCELSO.set_title("Tipos de SO de los dispositivos donde se origino el evento", fontsize=18)
plotCELSO.set_xlabel("SO",fontsize=18)
plotCELSO.set_ylabel("Cantidad de eventos", fontsize=18)
dfFiltradoSO['operating_system_version'].value_counts()[0:15]
pd.crosstab(df.operating_system_version,df.event)
#Se detallan los tipos de eventos, en este caso, solo visited site se encuentra en la lista.
#Tipo de canal donde se origino el evento
dfFiltradoCH = df.loc[df['channel'] != 'NaN', :]
dfFiltradoCH = dfFiltradoCH.loc[df['channel'] != 'Unknown', :]
plotCH=dfFiltradoCH['channel'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));
plotCH.set_title("Tipos de channel donde se origino el evento", fontsize=18)
plotCH.set_xlabel("channel",fontsize=18)
plotCH.set_ylabel("Cantidad de eventos", fontsize=18)

dfFiltradoCH['channel'].value_counts()[0:19]
pd.crosstab(df.channel,df.event)
#Se detalla el tipo de evento , en este caso es para visitar el sitio
#Si es que viene de un motor de busqueda, ¿de cual?

dfFiltradoSE = df.loc[df['search_engine'] != 'NaN', :]
dfFiltradoSE = dfFiltradoSE.loc[df['search_engine'] != 'Unknown', :]
plotSE=dfFiltradoSE ['search_engine'].value_counts()[0:19].plot(kind='bar',figsize=(14,4));
plotSE.set_title("Tipos de motor de busqueda (si es que existe) donde se origino el evento", fontsize=18)
plotSE.set_xlabel("Motor de busqueda",fontsize=18)
plotSE.set_ylabel("Cantidad de eventos", fontsize=18)
plotSE.set_yscale('log')
pd.crosstab(df.search_engine,df.event)
#podemos ver que el usuario ingresa al sitio mediante un motor de búsqueda web.
#se detallan a continuacion los motores de busqueda contra los tipos de eventos.

#tipo de navegador usado en el evento

dfFiltradoBV = df.loc[df['browser_version'] != 'NaN', :]
dfFiltradoBV = dfFiltradoBV.loc[df['browser_version'] != 'Unknown', :]
plotBV=dfFiltradoBV ['browser_version'].value_counts()[0:30].plot(kind='bar',figsize=(14,4));
plotBV.set_title("Tipos de navegador usado en el evento", fontsize=18)
plotBV.set_xlabel("Navegador",fontsize=18)
plotBV.set_ylabel("Cantidad de eventos", fontsize=18)

dfFiltradoBV ['browser_version'].value_counts()[0:30]
pd.crosstab(df.browser_version,df.event)
#diferentes tipos de eventos x (visited site) contra navegador usado
