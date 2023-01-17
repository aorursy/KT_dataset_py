import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
viajes = pd.read_csv("../input/sf-bay-area-bike-share/trip.csv", 

                     parse_dates=["start_date", "end_date"], infer_datetime_format=True,

                     dtype={"subscription_type":"category"})

viajes.head(1)
viajes.dtypes
viajes.duration.sort_values().head(5)
viajes.duration.sort_values().tail(10)/(3600*24)
cuantilDuracion = viajes.duration.quantile(0.995)

cuantilDuracion /3600
viajes_sin_outliers = viajes[viajes.duration < cuantilDuracion]
viajes_sin_outliers.duration.apply(lambda x: x/3600).plot.box();

plt.title("Duraciones de los recorridos.");

plt.show();
print("Media:   " + str(viajes_sin_outliers.duration.mean()))

print("Mediana: " + str(viajes_sin_outliers.duration.median()))
# Distribucion de la duracion de los viajes en cantidad de minutos

viajes.duration.apply(lambda d: d/100).hist(bins=range(1,100));

plt.show()
# Haciendo foco en los de menos de 20 minutos

viajes.duration.hist(bins=range(1,1200,10));

plt.show()
# Agrego el week_day

dias_ordenados = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

viajes["week_day"] = pd.Categorical(viajes.start_date.dt.weekday_name, categories=dias_ordenados)
# Cantidad de días distintos

fechasDistintas = pd.DataFrame({"fecha": viajes.start_date.dt.date, "week_day": viajes.week_day})

cantDias = fechasDistintas.groupby(["week_day","fecha"]).agg(lambda x:1).reset_index().week_day.value_counts()

# Arreglo del índice categórico.

cantDias.index = pd.Categorical(cantDias.index, categories=dias_ordenados)

cantDias = cantDias.sort_index()

cantDias
totalDias = cantDias.sum()

totalDias
(viajes.groupby("week_day").size()/cantDias).plot(kind='bar',figsize=(12,8));

plt.title("Cantidad de alquileres promedio segun dia de la semana")

plt.xlabel("Dia de la semana")

plt.ylabel("Cantidad de alquileres")

plt.show()
viajes["finde"] = viajes.start_date.dt.dayofweek >= 5
# Días de semana(en minutos)

viajes.loc[~viajes.finde,"duration"].apply(lambda x: x/60).hist(bins=range(1,100));

plt.title("Distribucion de duraciones en la semana")

plt.xlabel("Minutos")

plt.ylabel("Apariciones")

plt.show()
# Fines de semana (en minutos)

viajes.loc[viajes.finde,"duration"].apply(lambda x: x/60).hist(bins=range(1,100));

plt.title("Distribucion de duraciones en los fines de semana")

plt.xlabel("Minutos")

plt.ylabel("Apariciones")

plt.show()
viajes["hour"] = viajes.start_date.dt.hour
# Días de semana

viajes.loc[~viajes.finde, "hour"].hist(bins=range(0,23));

plt.title("Cantidad de Alquileres por Hora en Días de semana")

plt.xlabel("Hora")

plt.show()
# Fines de semana

viajes.loc[viajes.finde, "hour"].hist(bins=range(0,23));

plt.title("Cantidad de Alquileres para fines de semana")

plt.xlabel("Hora")

plt.show()
viajes.loc[viajes.finde,['duration', 'hour']].groupby('hour').mean().plot.bar();

plt.title("Duración promedio en fines de semana.")

plt.xlabel("Hora")

plt.ylabel("Duración en segundos")

plt.show()
viajes.loc[~viajes.finde,['duration', 'hour']].groupby('hour').mean().plot.bar();

plt.title("Duración promedio en días laborales.")

plt.xlabel("Hora")

plt.ylabel("Duración en segundos")

plt.show()
estaciones = pd.read_csv("../input/sf-bay-area-bike-share/station.csv")
#armo un diccionario de distancias



ids = estaciones.id

stm = estaciones[["id", "lat", "long"]]



distancias = {}

for id1 in ids:

    x = stm.loc[estaciones.id == id1,["lat","long"]].values

    x = (x[0][0],x[0][1])

    for id2 in ids:

        y = stm.loc[estaciones.id == id2,["lat","long"]].values

        y = (y[0][0],y[0][1])

        # Distancia Manhattan

        distancias[(id1,id2)] = abs(y[0]-x[0]) + abs(y[1] - x[1])
def d(id1,id2):

    return distancias[(id1,id2)]
viajes["dist"] = viajes.apply(lambda x: d(x.start_station_id, x.end_station_id), axis=1)
# Radio de la tierra en KM

from math import pi

from numpy import sin



R = 6378.137

# Conversor de grados a radianes

conv = pi/180



# Conversión a km de toda la columna.

viajes["distkm"] = R*sin(conv*viajes.dist)
viajes[["distkm","duration"]].plot.scatter(x="distkm",y="duration");

plt.title("Duración según distancia recorrida")

plt.xlabel("Distancia (km)")

plt.ylabel("Duración (s)")

plt.show()
viajesCortos = viajes[viajes.duration < viajes.duration.quantile(0.9)]

viajes.duration.quantile(0.9)/60
viajesCortos.distkm.describe()
viajesCortos[["distkm","duration"]].plot.scatter(x="distkm",y="duration", alpha=0.05);

plt.title("Duración según distancias en viajes cortos")

plt.xlim([0,9])

plt.show()
# Viajes de menos de 50 minutos.

viajesMedianos = viajes[viajes.duration < 3000]
viajesMedianos[["distkm","duration"]].plot.scatter(x="distkm",y="duration", alpha=0.05);

plt.title("Duración según distancias en viajes medianos")

plt.xlim([0,9])

plt.show()
sns.heatmap(viajesCortos.corr().abs());

plt.title("Heatmap de correlación")

plt.show()
sns.heatmap(viajesMedianos.corr().abs());

plt.title("Heatmap de correlación para viajes hasta medianos")

plt.show()