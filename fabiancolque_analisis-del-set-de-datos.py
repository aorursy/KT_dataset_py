# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# magic function para hacer que los graficos de matplotlib se renderizen en el notebook.

%matplotlib inline



import datetime as datetime

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('default') # Make the graphs a bit prettier

plt.rcParams['figure.figsize'] = (15, 5)
#Cargo los datos pero parseando las fechas a DataTime

trip = pd.read_csv('../input/trip.csv', parse_dates=['start_date','end_date'])
#Observacion de los datos

trip.head(10)
#A cada dato de la columna de comienzo del viaje (start_date) le aplico una funcion para saber en que dia de la semana fueron

#realizados los viajes

#Aclaracion: dayofweek nos da los dias ordenados desde 0(lunes) hasta 6(domingo)

#Realizo un plot de barras para visualizar lo calculado en el paso anterior

plt = trip['start_date'].apply(lambda x: x.dayofweek).value_counts().plot('bar',figsize=(8,3));

plt.set_xlabel('Dias de la semana');

plt.set_ylabel('Cantidad');

plt.set_title('Cantidad de viajes por dia de la semana');

plt.set_xticklabels(['Martes','Miercoles','Jueves','Lunes','Viernes','Sabado','Domingo'], fontdict=None, minor=False);
#A cada dato de la columna de comienzo del viaje (start_date) le aplico una funcion para saber en que mes fueron

#realizados los viajes

#Realizo un plot en el cual observamos la cantidad de viajes segun el mes del año

plt = trip['start_date'].apply(lambda x: x.month).value_counts().plot('bar',figsize=(8,3));

plt.set_xlabel('Meses');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Cantidad de viajes por mes');
#Ahora para hacer un visualizacion de todos los viajes a traves del tiempo creamos una nueva columa en la cual tendremos

#la fecha pero sin la hora ni los minutos

trip['start_date_without_time']=trip.start_date.dt.date

trip.head(10)
#Realizamos una visualizacion de los viajes a traves del tiempo

#Quiero aclarar que se realizo una agrupacion dia a dia para realizar este plot

plt = trip.groupby('start_date_without_time').count()['id'].plot(figsize=(8,3));

plt.set_xlabel('Fecha');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Cantidad de viajes a traves del tiempo');
#Ahora añadiremos otra columna la cual tendra solo la hora en la que se realiza el viaje

trip['hora'] = trip['start_date'].apply(lambda x: x.hour)

#Realizo una visualizacion en base a la hora en que se realiza el viaje

plt = trip.groupby('hora').count()['id'].plot('bar',figsize=(8,3));

plt.set_xlabel('Hora del dia');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Cantidad de viajes dependiendo de la hora');
#Cambiamos la duracion a minutos

trip['duration_min'] = trip['duration'].apply(lambda x: int(x/60))

#Cantidad de viajes segun la duracion (en minutos). Visualizacion de la cantidad de viajes segun la duracion del viaje 

plt = trip['duration_min'].value_counts().head(20).plot('bar',figsize=(8,3));

plt.set_xlabel('Cantidad de minutos del viaje');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Top 20 de cantidad de viajes dependiendo de su duracion');
#Cargamos los datos de station.csv y le cambiamos el nombre a una de sus columnas para un posterior procesamiento

station = pd.read_csv('../input/station.csv', low_memory=False)

station.rename(columns={'id': 'start_station_id'}, inplace=True)

station.head(10)
#Realizamos un join entre trip y station en base a la columna start_station_id

arch_unidos = pd.merge(trip, station, on='start_station_id', how='inner')

arch_unidos.head(10)
#Visulizacion de la cantidad de viajes segun la ciudad

plt = arch_unidos['city'].value_counts().plot('bar',figsize=(8,3));

plt.set_xlabel('Ciudad');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Cantidad de viajes dependiendo de la ciudad');
#Visualizacion de la cantidad de viajes segun la estacion

#Solo mostramos las 20 ciudades con menos cantidad de viajes

plt = arch_unidos['start_station_name'].value_counts().tail(20).plot('bar',figsize=(8,3))

plt.set_xlabel('Estacion de bicicleta')

plt.set_ylabel('Cantidad de viajes')

plt.set_title('Top20 de estaciones de salida con menos cantidad de viajes');
#Visualizacion de la cantidad de viajes segun la estacion

#Solo mostramos las 20 ciudades con mas de viajes

plt = arch_unidos['start_station_name'].value_counts().head(20).plot('bar',figsize=(8,3))

plt.set_xlabel('Estacion de bicicleta')

plt.set_ylabel('Cantidad de viajes')

plt.set_title('Top20 de estaciones de salida con mas cantidad de viajes');
#Visualizacion de la cantidad de viajes segun la estacion

#Solo mostramos las 20 ciudades con menos cantidad de viajes

plt = arch_unidos['end_station_name'].value_counts().tail(20).plot('bar',figsize=(8,3))

plt.set_xlabel('Estacion de bicicleta')

plt.set_ylabel('Cantidad de viajes')

plt.set_title('Top20 de estaciones de llegada con menos cantidad de viajes');
#Visualizacion de la cantidad de viajes segun la estacion

#Solo mostramos las 20 ciudades con mas de viajes

plt = arch_unidos['end_station_name'].value_counts().head(20).plot('bar',figsize=(8,3))

plt.set_xlabel('Estacion de bicicleta')

plt.set_ylabel('Cantidad de viajes')

plt.set_title('Top20 de estaciones de llegada con mas cantidad de viajes');
# Cantidad de viajes por bicicleta.

plt = trip.groupby('bike_id').count()['id'].plot(figsize=(9,3));

plt.set_xlabel('ID de Bicicleta')

plt.set_ylabel('Cantidad de viajes')

plt.set_title('Cantidad de viajes por bicicleta');
# Duracion de viajes por bicicleta.

plt = trip.groupby('bike_id').sum()['duration_min'].plot(figsize=(9,3));

plt.set_xlabel('ID de Bicicleta')

plt.set_ylabel('Duracion(minutos)')

plt.set_title('Duracion de viajes por bicicleta');
#Podemos ver por el grafico anterior que teniamos algunos datos anomalos, por ende filtramos los que tienen duracion

#menor a 5000 minutos

#Filtramos los viajes para que solo queden los de duracion menor a 5000

trip_con_duracion_filtrada = trip[trip['duration_min'] < 5000 ]

# Duracion de viajes por bicicleta.

plt = trip_con_duracion_filtrada.groupby('bike_id').sum()['duration_min'].plot(figsize=(9,3));

plt.set_xlabel('ID de Bicicleta')

plt.set_ylabel('Duracion (minutos)')

plt.set_title('Duracion de viajes por bicicleta');
#Ahora intentamos buscar de esos IDs a donde pertencen, o sea en que estacion de bicicleta se encontraban.

#Filtramos los IDs anteriores para saber a que ciudad pertenecen 

id_mayor_a_725 = arch_unidos['bike_id'] > 725

id_menor_a_880 = arch_unidos['bike_id'] < 880

arch_unidos_con_IDs_filtrados = arch_unidos[ id_mayor_a_725 & id_menor_a_880]

#Visulizacion de la cantidad de viajes segun la ciudad

plt = arch_unidos_con_IDs_filtrados['start_station_name'].value_counts().head(20).plot('bar',figsize=(9,3))

plt.set_xlabel('Estacion de bicicleta');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Top20 de estaciones de salida con mas cantidad de viajes(solo para IDs mayores a 725 y menores a 880)');
# se carga weather, y la columna start_date de trip para extraer la fecha sin hora en un string.

weather = pd.read_csv('../input/weather.csv')

start_date = pd.read_csv('../input/trip.csv', usecols = [2])

trip['date'] = start_date['start_date'].str.extract(r'(\d+/\d+/\d+)',expand=False)
#Se reducen la cantidad de columnas de los archivos por una cuestion de memoria

trip_para_mergear= trip[['id','date']]

weather_para_mergear = weather[['max_temperature_f','date','mean_temperature_f','mean_humidity']]

# se unen trip y weather a partir de date.

tw = pd.merge(trip_para_mergear, weather_para_mergear, left_on='date', right_on='date', how='inner')
#calcula promedio de temperatura máxima por día y cuenta la cantidad de viajes de ese día. Ordena por temeperatura. 

grouped = tw.loc[:,['date','max_temperature_f']].groupby('date').agg(['mean','count']).max_temperature_f.sort_values('mean',ascending=True)

grouped.reset_index('date')

plot = grouped.plot(figsize=(9,3),x = 'mean', y = 'count', kind = 'scatter', title='Cantidad de viajes dependiendo la temperatura maxima')

plot.set_ylabel('Cantidad de viajes');

plot.set_xlabel('Temperatura maxima promedio(Fahrenheit)');



# Cuando la temperatura máxima es menor a 60 F, se nota una clara disminución en la cantidad de viajes.

# Con temperaturas mayores, la cantidad de vaijes se mantiene.

# Disminuyen con temperaturas maximas mayores a 90 F.
#calcula promedio de temperatura por día y cuenta la cantidad de viajes de ese día. Ordena por temeperatura. 

grouped = tw.loc[:,['date','mean_temperature_f']].groupby('date').agg(['mean','count']).mean_temperature_f.sort_values('mean',ascending=True)

grouped.reset_index('date')

plot = grouped.plot(figsize=(9,3),x = 'mean', y = 'count', kind = 'scatter', title='Cantidad de viajes dependiendo la temperatura promedio')

plot.set_ylabel('Cantidad de viajes');

plot.set_xlabel('Temperatura promedio (Fahrenheit)');
#calcula promedio de temperatura por día y cuenta la cantidad de viajes de ese día. Ordena por temeperatura. 

grouped = tw.loc[:,['date','mean_humidity']].groupby('date').agg(['mean','count']).mean_humidity.sort_values('mean',ascending=True)

grouped.reset_index('date')

plot = grouped.plot(figsize=(9,3),x = 'mean', y = 'count', kind = 'scatter', title='Cantidad de viajes dependiendo de la humedad promedio')

plot.set_ylabel('Cantidad de viajes');

plot.set_xlabel('Humedad promedio');
#Cantidad de viajes segun la duracion (en minutos). Visualizacion de la cantidad de viajes segun la duracion del viaje 

plt = trip['subscription_type'].value_counts().plot('bar',figsize=(9,3));

plt.set_xlabel('Suscriptor');

plt.set_ylabel('Cantidad de viajes');

plt.set_title('Cantidad de viajes dependiendo del suscriptor');
#Comparacion de viajes por día de la semana dependiendo el tipo de subscripcion.

tripc_plot = trip.loc[trip.subscription_type == 'Customer',['start_date']]['start_date'].apply(lambda x: datetime.datetime.strftime(x, '%A')).value_counts().plot('bar',figsize=(9,3), color='blue', position = 0, width= 0.4);

trip.loc[trip.subscription_type == 'Subscriber',['start_date']]['start_date'].apply(lambda x: datetime.datetime.strftime(x, '%A')).value_counts().plot('bar',figsize=(9,3),color='red', position = 1, width= 0.4);



#En la visualizacion tenemos en rojo los suscriptores y en azul los clientes comunes

tripc_plot.set_xlabel('Dias de la semana');

tripc_plot.set_ylabel('Cantidad de viajes');

tripc_plot.set_title('Cantidad de viajes por dia de la semana');