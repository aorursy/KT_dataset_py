#link del repositorio en github: https://github.com/guidoAlbarello/DatosDatosos



%matplotlib inline



import datetime as dt

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('default') # Make the graphs a bit prettier

plt.rcParams['figure.figsize'] = (15, 5)
#levantamos los csv



station = pd.read_csv('station.csv', low_memory=False)

trip_test = pd.read_csv('trip_test.csv', low_memory=False)

trip_train = pd.read_csv('trip_train.csv', low_memory=False)

weather = pd.read_csv('weather.csv', low_memory=False)
status = pd.read_csv('status.csv', low_memory=True)
#vemos un poco los dataframes de cada csv

station.head()
status.head()
trip_train.head(10)
weather.head()
#vemos de que años son los registros   (2013, 2014,2015)  tal vez lo dice en kaggle pero por las dudas



print(trip_train['start_date'].str.contains('2011').value_counts())

print(trip_train['start_date'].str.contains('2012').value_counts())

print(trip_train['start_date'].str.contains('2013').value_counts())

print(trip_train['start_date'].str.contains('2014').value_counts())

print(trip_train['start_date'].str.contains('2015').value_counts())

print(trip_train['start_date'].str.contains('2016').value_counts())
#promedio de duracion de viajes para cada año



sum = 0

count = trip_train[trip_train['start_date'].str.contains('2013')]['duration'].count()

idx =  (trip_train[trip_train['start_date'].str.contains('2013')])['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['start_date'].str.contains('2013')]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioAnio2013 = sum / count



sum = 0

count = trip_train[trip_train['start_date'].str.contains('2014')]['duration'].count()

idx =  (trip_train[trip_train['start_date'].str.contains('2014')])['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['start_date'].str.contains('2014')]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioAnio2014 = sum / count



sum = 0

count = trip_train[trip_train['start_date'].str.contains('2015')]['duration'].count()

idx =  (trip_train[trip_train['start_date'].str.contains('2015')])['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['start_date'].str.contains('2015')]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioAnio2015 = sum / count



print(promedioAnio2013)

print(promedioAnio2014)

print(promedioAnio2015)
#graficamos el prmedio por año de duracion de viajes 

%matplotlib notebook

promediosAnios = [promedioAnio2013, promedioAnio2014, promedioAnio2015]

anios = [2013,2014,2015]

plt.bar(anios, promediosAnios, width = 0.4)

plt.suptitle("Duracion de viaje promedio por año")

plt.xlabel("Años")

plt.ylabel("Duracion promedio de viaje (s)")

plt.xticks(anios, ['2013','2014', '2015'])

plt.show()
#analizamos en que influye el ser costumer o subscriber respecto al uso de bicis o duration de viaje



trip_train['subscription_type'].value_counts()

sum = 0

count = trip_train[trip_train['subscription_type'] == 'Subscriber']['duration'].count()

idx =  trip_train[trip_train['subscription_type'] == 'Subscriber']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['subscription_type'] == 'Subscriber']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSubscriber = sum / count



sum = 0

count = trip_train[trip_train['subscription_type'] == 'Customer']['duration'].count()

idx =  trip_train[trip_train['subscription_type'] == 'Customer']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['subscription_type'] == 'Customer']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioCustomer = sum / count



#tla vez aca alla problema de la ecuacion de moivre 

print(promedioSubscriber)

print(promedioCustomer)
#calculamos duracion promedio por dia de semana



#separamos la fecha por / y eliminamos todo lo q hay mas a la derecha del espacio despues del año

fechas = trip_train['start_date'].map(lambda x: (x.split('/')[0], x.split('/')[1], x.split('/')[2].split(' ')[0]))



#convertimos cada tupla de año mes dia a dia de semana ylo guardamos en un nuevo campo 

trip_train['weekday'] = fechas.map(lambda x: dt.date(int(x[2]),int(x[0]),int(x[1])).strftime("%A"))
#grafico de viajes q se realizan cada dia de semana 

%matplotlib notebook



weekday_count = trip_train['weekday'].value_counts()

number = [0,1,2,3,4,5,6]

dias = ['Tue', 'Wed', 'Thu', 'Mon', 'Fri', 'Sat', 'Sun']

plt.bar(number, weekday_count)

plt.suptitle("Cantidad de viajes según el día de semana")

plt.xlabel("Día de la semana")

plt.ylabel("Cantidad de viajes")

plt.xticks(number, dias)

plt.show()
#duracion por dia de semana





sum = 0

count = trip_train[trip_train['weekday'] == 'Monday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Monday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Monday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioLunes = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Tuesday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Tuesday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Tuesday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMartes = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Wednesday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Wednesday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Wednesday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMiercoles = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Thursday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Thursday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Thursday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioJueves = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Friday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Friday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Friday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioViernes = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Saturday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Saturday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Saturday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSabado = sum / count



sum = 0

count = trip_train[trip_train['weekday'] == 'Sunday']['duration'].count()

idx =  trip_train[trip_train['weekday'] == 'Sunday']['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['weekday'] == 'Sunday']['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioDomingo = sum / count

                       

print(promedioLunes)

print(promedioMartes)

print(promedioMiercoles)

print(promedioJueves)

print(promedioViernes)

print(promedioSabado)

print(promedioDomingo)             #habria q chequear estos datos al parecer los findes se viaja mas q el dia de semana. 

                                   #aunq tendria sentido pensando que en la semana se usa mucho para trabajar y el finde para pasear

                                   #y el viernes es mas alto porq se junta trabajo y un poquito de paseo

                       
#graficamos los promedios de arriba 

%matplotlib notebook

promediosDias = [promedioLunes, promedioMartes, promedioMiercoles, promedioJueves, promedioViernes, promedioSabado, promedioDomingo]

number = [0,1,2,3,4,5,6]

dias = [ 'Mon', 'Tue', 'Wed', 'Thu','Fri', 'Sat', 'Sun']

plt.bar(number, promediosDias)

plt.suptitle("Duracion de viaje promedio según el día de semana")

plt.xlabel("Día de la semana")

plt.ylabel("Duracion promedio de viajes (s)")

plt.xticks(number, dias)

plt.show()
#status[(status['docks_available'] == 0)]['station_id'].value_counts()
cantidadVecesConCeroBikesDisponibles = [1] * station['id'].size

j = 0

for stationId in status['station_id'].unique():

    statusConId = status[status['station_id'] == stationId]

    cantidadVecesConCeroBikesDisponibles[j] = statusConId[statusConId['bikes_available'] == 0]['station_id'].count()

    j = j + 1



station['cantidad_veces_cero_bikes_disponibles'] = cantidadVecesConCeroBikesDisponibles



cantidadVecesConCeroDocksDisponibles = [1] * station['id'].size

j = 0

for stationId in status['station_id'].unique():

    statusConId = status[status['station_id'] == stationId]

    cantidadVecesConCeroDocksDisponibles[j] = statusConId[statusConId['docks_available'] == 0]['station_id'].count()

    j = j + 1



station['cantidad_veces_cero_docks_disponibles'] = cantidadVecesConCeroDocksDisponibles
promedioBikesAvailable = [1] * station['id'].size

j = 0

for stationId in status['station_id'].unique():

    sum = 0

    count = status[status['station_id'] == stationId]['bikes_available'].count()

    idx =  status[status['station_id'] == stationId]['bikes_available'].value_counts().index.tolist()

    i = 0

    availability = status[status['station_id'] == stationId]['bikes_available'].value_counts()

    for bikesAvailable in availability:

        sum += idx[i] * bikesAvailable

        i += 1

    promedioBikesAvailable[j] = round(sum/count , 2)

    j = j + 1

station['promedio_bikes_available'] = promedioBikesAvailable





promedioDocksAvailable = [1] * station['id'].size

j = 0

for stationId in status['station_id'].unique():

    sum = 0

    count = status[status['station_id'] == stationId]['docks_available'].count()

    idx =  status[status['station_id'] == stationId]['docks_available'].value_counts().index.tolist()

    i = 0

    availability = status[status['station_id'] == stationId]['docks_available'].value_counts()

    for docksAvailable in availability:

        sum += idx[i] * docksAvailable

        i += 1

    promedioDocksAvailable[j] = round(sum/count , 2)

    j = j + 1

station['promedio_docks_available'] = promedioDocksAvailable
station.head()
#grafico de todas las estaciones con radiosegun el promedio de las bicicletas disponibles



#mejorar los colores, creando una distancia entre cero y uno 



%matplotlib notebook



plt.scatter(station['lat'], station['long'], s=station['promedio_bikes_available']*5)



plt.suptitle("Estaciones según el promedio de la cantidad de bicicletas disponibles")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del primer conjunto



#suponemos un criterio para poca, media y mucha cantidad de bicis disponibles



%matplotlib notebook



lat37 = station[station['lat'] < 37.37]



colors = ["" for x in range(lat37['promedio_bikes_available'].size)]

i = 0

for line in lat37['promedio_bikes_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1

plt.scatter(lat37['lat'], lat37['long'], s=lat37['promedio_bikes_available']*20, c = colors )



plt.suptitle("Estaciones según el promedio de la cantidad de bicicletas disponibles (1er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()

#aregar q si son de un radio menor a x , sean de otro color. tres colores distintos. poco, medio y mucha disponibilidad
#grafico de todas las estaciones del segundo conjunto



%matplotlib notebook



lat36 = station[(station['lat'] > 37.37) & (station['lat'] < 37.75)]

colors = ["" for x in range(lat36['promedio_bikes_available'].size)]

i = 0

for line in lat36['promedio_bikes_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1



plt.scatter(lat36['lat'], lat36['long'], s=lat36['promedio_bikes_available']*20, c = colors)

plt.suptitle("Estaciones según el promedio de la cantidad de bicicletas disponibles (2do conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del tercer conjunto



%matplotlib notebook



lat75 = station[station['lat'] > 37.75]

colors = ["" for x in range(lat75['promedio_bikes_available'].size)]

i = 0

for line in lat75['promedio_bikes_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1



plt.scatter(lat75['lat'], lat75['long'], s=lat75['promedio_bikes_available']*20, c = colors)

plt.suptitle("Estaciones según el promedio de la cantidad de bicicletas disponibles (3er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones con radio segun el promedio de los docks disponibles



#falta emprolijarlos. agregar labels titulos



%matplotlib notebook



plt.scatter(station['lat'], station['long'], s=station['promedio_docks_available']*5)

plt.suptitle("Estaciones según el promedio de la cantidad de docks disponibles")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del primer conjunto



%matplotlib notebook



lat37 = station[station['lat'] < 37.37]

colors = ["" for x in range(lat37['promedio_docks_available'].size)]

i = 0

for line in lat37['promedio_docks_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1



plt.scatter(lat37['lat'], lat37['long'], s=lat37['promedio_docks_available']*20, c = colors)

plt.suptitle("Estaciones según el promedio de la cantidad de docks disponibles (1er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del segundo conjunto



%matplotlib notebook



lat36 = station[(station['lat'] > 37.37) & (station['lat'] < 37.75)]

colors = ["" for x in range(lat36['promedio_docks_available'].size)]

i = 0

for line in lat36['promedio_docks_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1



plt.scatter(lat36['lat'], lat36['long'], s=lat36['promedio_docks_available']*20, c = colors)

plt.suptitle("Estaciones según el promedio de la cantidad de docks disponibles (2do conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del tercer conjunto



%matplotlib notebook



lat75 = station[station['lat'] > 37.75]

colors = ["" for x in range(lat75['promedio_docks_available'].size)]

i = 0

for line in lat75['promedio_docks_available']:

    if(int(line) < 3):

        colors[i] = 'red'

    elif((int(line) > 3) & (int(line) < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'green'

    i = i +1



plt.scatter(lat75['lat'], lat75['long'], s=lat75['promedio_docks_available']*20, c = colors)

plt.suptitle("Estaciones según el promedio de la cantidad de docks disponibles (3er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones con radio segun cantidad de veces q se quedaron sin bicis

#falta emprolijarlos. agregar labels titulos



%matplotlib notebook



plt.scatter(station['lat'], station['long'], s=station['cantidad_veces_cero_bikes_disponibles']/200)

plt.suptitle("Estaciones según el tiempo que se quedaron sin bicicletas")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del primer conjunto



%matplotlib notebook



lat37 = station[station['lat'] < 37.37]

colors = ["" for x in range(lat37['cantidad_veces_cero_bikes_disponibles'].size)]

i = 0

for line in lat37['cantidad_veces_cero_bikes_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1



plt.scatter(lat37['lat'], lat37['long'], s=lat37['cantidad_veces_cero_bikes_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin bicicletas (1er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del segundo conjunto



%matplotlib notebook



lat36 = station[(station['lat'] > 37.37) & (station['lat'] < 37.75)]

colors = ["" for x in range(lat36['cantidad_veces_cero_bikes_disponibles'].size)]

i = 0

for line in lat36['cantidad_veces_cero_bikes_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1

    

plt.scatter(lat36['lat'], lat36['long'], s=lat36['cantidad_veces_cero_bikes_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin bicicletas (2do conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
%matplotlib notebook



lat75 = station[station['lat'] > 37.75]

colors = ["" for x in range(lat75['cantidad_veces_cero_bikes_disponibles'].size)]

i = 0

for line in lat75['cantidad_veces_cero_bikes_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1

    

plt.scatter(lat75['lat'], lat75['long'], s=lat75['cantidad_veces_cero_bikes_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin bicicletas (3er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones con radio segun la cantidad de veces que se quedaron sin docks



#falta emprolijarlos. agregar labels titulos



%matplotlib notebook



plt.scatter(station['lat'], station['long'], s=station['cantidad_veces_cero_docks_disponibles']/200)

plt.suptitle("Estaciones según el tiempo que se quedaron sin docks")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del primer conjunto



%matplotlib notebook



lat37 = station[station['lat'] < 37.37]

colors = ["" for x in range(lat37['cantidad_veces_cero_docks_disponibles'].size)]

i = 0

for line in lat37['cantidad_veces_cero_docks_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1



plt.scatter(lat37['lat'], lat37['long'], s=lat37['cantidad_veces_cero_docks_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin docks (1er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#grafico de todas las estaciones del segundo conjunto



%matplotlib notebook



lat36 = station[(station['lat'] > 37.37) & (station['lat'] < 37.75)]

colors = ["" for x in range(lat36['cantidad_veces_cero_docks_disponibles'].size)]

i = 0

for line in lat36['cantidad_veces_cero_docks_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1

plt.scatter(lat36['lat'], lat36['long'], s=lat36['cantidad_veces_cero_docks_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin docks (2do conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
%matplotlib notebook



lat75 = station[station['lat'] > 37.75]

colors = ["" for x in range(lat75['cantidad_veces_cero_docks_disponibles'].size)]

i = 0

for line in lat75['cantidad_veces_cero_docks_disponibles']:

    if(int(line)/1000 < 3):

        colors[i] = 'green'

    elif((int(line)/1000 > 3) & (int(line)/1000 < 8)):

        colors[i] = 'orange'

    else:

        colors[i] = 'red'

    i = i +1



plt.scatter(lat75['lat'], lat75['long'], s=lat75['cantidad_veces_cero_docks_disponibles']/50, c = colors)

plt.suptitle("Estaciones según el tiempo que se quedaron sin docks (3er conjunto)")

plt.xlabel("Latitud")

plt.ylabel("Longitud")

plt.show()
#ver dia y noche como afecta



#separamos la fecha por / y eliminamos todo lo q hay mas a la derecha del espacio despues del año

fechas = trip_train['start_date'].map(lambda x: (x.split('/')[0], x.split('/')[1], x.split('/')[2].split(' ')[0]))



horas = trip_train['start_date'].map(lambda x:x.split('/')[2].split(' ')[1].split(':')[0])

#convertimos cada tupla de año mes dia a dia de semana ylo guardamos en un nuevo campo 



fechas = fechas.map(lambda x: dt.date(int(x[2]),int(x[0]),int(x[1])).strftime("%m"))



i = 0

deNoche = [0] * fechas.size

for line in fechas:

    if(((int(line) < 4) & (int(line) > 0)) | ((int(line) > 9)&(int(line) < 13))):

        if((int(horas[i]) <18) & (int(horas[i]) > 7)):

            deNoche[i] = 0

        else:

            deNoche[i] = 1

    else:

        if((int(horas[i]) <20) & (int(horas[i]) > 6)):

            deNoche[i] = 0

        else:

            deNoche[i] = 1

    i = i + 1 

        

trip_train['deNoche'] = deNoche

#cantidad viajes de noche y dia

trip_train['deNoche'].value_counts()
#duracion viajes de noche y dia





sum = 0

count = trip_train[trip_train['deNoche'] == 0]['duration'].count()

idx =  trip_train[trip_train['deNoche'] == 0]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['deNoche'] == 0]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioDia = sum / count



sum = 0

count = trip_train[trip_train['deNoche'] == 1]['duration'].count()

idx =  trip_train[trip_train['deNoche'] == 1]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['deNoche'] == 1]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioNoche = sum / count





print (promedioDia)

print (promedioNoche)
#duracion viajes de dia en dias de semana



sum = 0

count = trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioLunesDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMartesDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMiercolesDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioJuevesDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioViernesDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSabadoDia = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 0)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 0)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 0)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioDomingoDia = sum / count









#duracion viajes de noche en dias de semana



sum = 0

count = trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Monday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioLunesNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Tuesday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMartesNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Wednesday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioMiercolesNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Thursday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioJuevesNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Friday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioViernesNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Saturday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSabadoNoche = sum / count



sum = 0

count = trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 1)]['duration'].count()

idx =  trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 1)]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[(trip_train['weekday'] == 'Sunday') & (trip_train['deNoche'] == 1)]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioDomingoNoche = sum / count
%matplotlib notebook

promediosDias = [promedioLunesDia, promedioMartesDia, promedioMiercolesDia, promedioJuevesDia, promedioViernesDia, promedioSabadoDia, promedioDomingoDia]

promediosNoches = [promedioLunesNoche, promedioMartesNoche, promedioMiercolesNoche, promedioJuevesNoche, promedioViernesNoche, promedioSabadoNoche, promedioDomingoNoche]



number = [0,1,2,3,4,5,6]

dias = [ 'Mon', 'Tue', 'Wed', 'Thu','Fri', 'Sat', 'Sun']

plt.bar(number, promediosNoches, width = 0.4)

plt.bar(number, promediosDias, width = 0.6, align = 'edge')

plt.suptitle("Duración promedio de día y noche según el día de semana")

plt.xlabel("Duración de viaje (s)")

plt.ylabel("Día de la semana")

plt.xticks(number, dias)

plt.show()
#analizar segun las estaciones del año







#separamos la fecha por / y eliminamos todo lo q hay mas a la derecha del espacio despues del año

fechas = trip_train['start_date'].map(lambda x: (x.split('/')[0], x.split('/')[1], x.split('/')[2].split(' ')[0]))



fechas = fechas.map(lambda x: dt.date(int(x[2]),int(x[0]),int(x[1])).strftime("%m%d"))



i = 0

estacion = [0] * fechas.size

for line in fechas:

    if((int(line) < 621) & (int(line) >= 321)):

        estacion[i] = 0

    elif((int(line) < 921) & (int(line) >= 621)):

        estacion[i] = 1

    elif((int(line) < 1221) & (int(line) >= 921)):

        estacion[i] = 2

    else:

        estacion[i] = 3

    i = i + 1 

        

trip_train['estacion'] = estacion
sum = 0

count = trip_train[trip_train['estacion'] == 0]['duration'].count()

idx =  trip_train[trip_train['estacion'] == 0]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['estacion'] == 0]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSpring = sum / count



sum = 0

count = trip_train[trip_train['estacion'] == 1]['duration'].count()

idx =  trip_train[trip_train['estacion'] == 1]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['estacion'] == 1]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioSummer = sum / count



sum = 0

count = trip_train[trip_train['estacion'] == 2]['duration'].count()

idx =  trip_train[trip_train['estacion'] == 2]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['estacion'] == 2]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioAutumn = sum / count



sum = 0

count = trip_train[trip_train['estacion'] == 3]['duration'].count()

idx =  trip_train[trip_train['estacion'] == 3]['duration'].value_counts().index.tolist()

i = 0

durations = trip_train[trip_train['estacion'] == 3]['duration'].value_counts()

for duration in durations:

    sum += idx[i] * duration

    i += 1

promedioWinter = sum / count
#cantidad viajes 



print(trip_train[trip_train['estacion'] == 0]['duration'].count())

print(trip_train[trip_train['estacion'] == 1]['duration'].count())

print(trip_train[trip_train['estacion'] == 2]['duration'].count())

print(trip_train[trip_train['estacion'] == 3]['duration'].count())
%matplotlib notebook



promediosEstaciones = [promedioSpring, promedioSummer, promedioAutumn, promedioWinter]



number = [0,1,2,3]

estacionesDelAnio = [ 'Spr', 'Sum', 'Aut', 'Win']

plt.bar(number, promediosEstaciones, width = 0.2)

plt.suptitle("Duración de viajes según la estación del año")

plt.xlabel("Duración de viaje (s)")

plt.ylabel("Estación del año")

plt.xticks(number, estacionesDelAnio)

plt.show()
#averiguamos la cantidad de alquiler de bicis por estacion

alquiler_de_bicis_por_estacion = pd.DataFrame({'alquiler_de_bicis_por_estacion' : trip_train['start_station_name'].value_counts()})

alquiler_de_bicis_por_estacion
#graficamos

alquiler_de_bicis_por_estacion.plot(kind='bar')
#viajes entre estaciones

viajes_entre_estaciones = trip_train[['start_station_name','end_station_name','duration']]

viajes_entre_estaciones
#tiempo total de los viajes realziados entre las estaciones a,b

duracion_de_viaje_entre_estaciones = viajes_entre_estaciones.groupby(['start_station_name', 'end_station_name']).sum().sort_values(by = 'duration',ascending= False)

duracion_de_viaje_entre_estaciones
estacion_como_inicio = pd.DataFrame({'cantidad_de_veces_como_inicio' : trip_train['start_station_name'].value_counts()}).reset_index().rename(columns={'index': 'estacion'})

estacion_como_inicio
estacion_como_final = pd.DataFrame({'cantidad_de_veces_como_final' : trip_train['end_station_name'].value_counts()}).reset_index().rename(columns={'index': 'estacion'})

estacion_como_final
usos_de_estacion = pd.merge(estacion_como_inicio, estacion_como_final, on='estacion')

usos_de_estacion.plot(kind="area", stacked=False)
usos_de_estacion


usos_de_estacion['usos_de_estacion'] = usos_de_estacion['cantidad_de_veces_como_inicio'] + usos_de_estacion['cantidad_de_veces_como_final']

usos_de_estacion.sort_values(by = 'usos_de_estacion', ascending = False)
#Veamos ahora si la cantidad de bicis que llegan y salen de las estaciones varía según las condiciones del clima

#Veamos qué sucede cuando las temperaturas son elevadas (mayores a 25 grados C)

weather['date'] = pd.to_datetime(weather['date'])

weather['date_without_time'] = weather.date.dt.date

trip_train['start_date'] = pd.to_datetime(trip_train['start_date'])

trip_train['start_date_without_time'] = trip_train.start_date.dt.datet

high_temps = weather[(weather['max_temperature_f']>77) & (weather['max_temperature_f']<102)]

trip_train.set_index('start_date_without_time')

high_temps.set_index('date_without_time')

high_temps_trip_train = pd.merge(high_temps,trip_train,how='inner',left_index=True, right_index=True)
#Estaciones donde se retiraron mayor cantidad de bicis los días de altas temperaturas

high_temps_trip_train['start_station_name'].value_counts()
#Estaciones donde se retiraron mayor cantidad de bicis los días de altas temperaturas (mayor a 25 grados C)

high_temps_bikes_take_out = pd.DataFrame({'cantidad de bicis retiradas en altas temperaturas' : high_temps_trip_train['start_station_name'].value_counts()}) 

high_temps_bikes_take_out.plot(kind='bar')
#Estaciones donde se dejaron mayor cantidad de bicis los días de altas temperaturas

high_temps_trip_train['end_station_name'].value_counts()
#Estaciones donde se dejaron mayor cantidad de bicis los días de altas temperaturas

high_temps_bikes_drop_off = pd.DataFrame({'cantidad de bicis dejadas en altas temperaturas' : high_temps_trip_train['end_station_name'].value_counts()}) 

high_temps_bikes_drop_off.plot(kind='bar')
#Veamos qué sucede cuando las temperaturas son bajas (menor a 25 grados)

low_temps = weather[(weather['max_temperature_f']<77)]

low_temps.set_index('date_without_time')

low_temps_trip_train = pd.merge(low_temps,trip_train,how='inner',left_index=True, right_index=True)

low_temps_trip_train
#Estaciones donde se retiraron mayor cantidad de bicis los días de temperaturas bajas

low_temps_trip_train['start_station_name'].value_counts()
#Estaciones donde se retiraron mayor cantidad de bicis los días de temperaturas bajas

low_temps_bikes_take_out = pd.DataFrame({'cantidad de bicis retiradas en bajas temperaturas' : low_temps_trip_train['start_station_name'].value_counts()}) 

low_temps_bikes_take_out.plot(kind='bar')
#Estaciones donde se dejaron mayor cantidad de bicis los días de temperaturas bajas

low_temps_trip_train['end_station_name'].value_counts()
#Estaciones donde se dejaron mayor cantidad de bicis los días de temperaturas bajas

low_temps_bikes_drop_off = pd.DataFrame({'cantidad de bicis dejadas en bajas temperaturas' : low_temps_trip_train['end_station_name'].value_counts()}) 

low_temps_bikes_drop_off.plot(kind='bar')
#Analicemos ahora para los días en los que hubo lluvias y en los que no llovió

#Veamos cuales son las estaciones con mayor movimiento de bicis dada esta condición climática

weather['precipitation_inches']=pd.to_numeric(weather['precipitation_inches'], errors='coerce')

rainy_weather = weather[(weather['precipitation_inches'] > 0) ]

rainy_weather.set_index('date_without_time')

rain_trip_train = pd.merge(rainy_weather,trip_train,how='inner',left_index=True, right_index=True)
#Estaciones donde se retiraron mayor cantidad de bicis los días de lluvias

rain_trip_train['start_station_name'].value_counts()
#Estaciones donde se retiraron mayor cantidad de bicis los días de lluvias

rain_trip_train_bikes_take_out = pd.DataFrame({'cantidad de bicis retiradas en días de lluvia' : rain_trip_train['start_station_name'].value_counts()}) 

rain_trip_train_bikes_take_out.plot(kind='bar')
#Estaciones donde se dejaron mayor cantidad de bicis los días sin lluvias

rain_trip_train['end_station_name'].value_counts()
#Estaciones donde se retiraron mayor cantidad de bicis los días de lluvias



rain_trip_train_bikes_drop_off = pd.DataFrame({'cantidad de bicis dejadas en días de lluvia' : rain_trip_train['end_station_name'].value_counts()}) 

rain_trip_train_bikes_drop_off.plot(kind='bar')
#Datos sin lluvia

#Veamos cuales son las estaciones con mayor movimiento de bicis 

no_rain_weather = weather[(weather['precipitation_inches'] == 0) ]

no_rain_weather.set_index('date_without_time')

no_rain_trip_train = pd.merge(rainy_weather,trip_train,how='inner',left_index=True, right_index=True)
#Estaciones donde se retiraron mayor cantidad de bicis los días sin lluvias

no_rain_trip_train['start_station_name'].value_counts()
#Estaciones donde se retiraron mayor cantidad de bicis los días de lluvias



no_rain_trip_train_bikes_take_out = pd.DataFrame({'cantidad de bicis retiradas en días de lluvia' : no_rain_trip_train['start_station_name'].value_counts()}) 

no_rain_trip_train_bikes_take_out.plot(kind='bar')
#Estaciones donde se dejaron mayor cantidad de bicis los días sin lluvias

no_rain_trip_train['end_station_name'].value_counts()
#Estaciones donde se dejaronmayor cantidad de bicis los días de lluvias

no_rain_trip_train_bikes_drop_of = pd.DataFrame({'cantidad de bicis dejadas en días de lluvia' : no_rain_trip_train['end_station_name'].value_counts()}) 

no_rain_trip_train_bikes_drop_of.plot(kind='bar')