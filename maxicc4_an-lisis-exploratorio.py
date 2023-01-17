import numpy as np 

import pandas as pd

import datetime

#import gmplot 



import seaborn as sns



from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay



import matplotlib.pyplot as plt

%matplotlib inline
stations = pd.read_csv("../input/station.csv")

stations.head()
print (str(len(stations.id.unique())))

print (str((stations.city.unique())))
print (stations.name.unique()) 

print (stations.id.unique())
print (str(stations.dock_count.sum()))
stations_installed = stations

stations_installed.installation_date = pd.to_datetime(stations_installed.installation_date, format='%m/%d/%Y') #Pase a formato datetime las fechas porque es mas comodo

stations_installed.plot(x='installation_date' , y = 'id', style='go',figsize=(14,5))



plt.show()
trips = pd.read_csv("../input/trip.csv")

trips.start_date = pd.to_datetime(trips.start_date, format='%m/%d/%Y %H:%M') #Pase a formato datetime las fechas porque es mas comodo

trips.end_date = pd.to_datetime(trips.end_date, format='%m/%d/%Y %H:%M')

trips.head()
print (str(len(trips.start_station_id.unique())))

print (str(len(trips.end_station_id.unique())))

print (str(len(trips.start_station_name.unique())))

print (str(len(trips.end_station_name.unique())))



print (str((trips.start_station_id.unique())))
# print (str(len(df.start_station_name.unique())))

stations_d = {}

for name in stations.name:

    stations_d[name] = 0;



stations_x = {}

for name in trips.start_station_name:

    if name not in stations_d:

        if name in stations_x:

            stations_x [name] += 1

        else:

            stations_x[name] = 0

stations_x = {}

for name in trips.end_station_name:

    if name not in stations_d:

        if name in stations_x:

            stations_x [name] += 1

        else:

            stations_x[name] = 0

        

        

print(stations_x)  


print ([x for x in  list(set(trips.start_station_name.unique())) if x not in list(set(stations.name.unique()))])



print ([x for x in  list(set(trips.start_station_id.unique())) if x not in list(set(stations.id.unique()))])



print (trips.loc[trips['start_station_name'] == 'San Jose Government Center','start_station_id'].unique(), trips.loc[trips['start_station_name'] == 'Washington at Kearny','start_station_id'].unique(), trips.loc[trips['start_station_name'] == 'Post at Kearny','start_station_id'].unique(), trips.loc[trips['start_station_name'] == 'Broadway at Main','start_station_id'].unique())
d={}

dok = {}



for id in stations.id.unique():   

    d[id] = list((set((trips.loc[trips['start_station_id'] == id, 'start_station_name']).tolist())))

    dok[id] = (stations.loc[stations['id'] == id, 'name']).to_string()



try:

    test =  pd.DataFrame(dok.items(), columns=['id','station_name'])

    test = test.set_index('id')

    test['trips_names'] = pd.Series(d, name='name')

except Exception:

    test =  pd.DataFrame()



test.head(70)
print (trips.loc[trips['start_station_name'] == 'San Jose Government Center','start_station_id'].count())

print (trips.loc[trips['end_station_name'] == 'San Jose Government Center','end_station_id'].count())

print (trips.loc[trips['start_station_name'] == 'Washington at Kearny','start_station_id'].count())

print (trips.loc[trips['end_station_name'] == 'Washington at Kearny','end_station_id'].count())

print (trips.loc[trips['start_station_name'] == 'Post at Kearny','start_station_id'].count())

print (trips.loc[trips['end_station_name'] == 'Post at Kearny','end_station_id'].count())

print (trips.loc[trips['start_station_name'] == 'Broadway at Main','start_station_id'].count())

print (trips.loc[trips['end_station_name'] == 'Broadway at Main','end_station_id'].count())
print (str(len(trips.zip_code.unique())))
trips_in_minute = trips

trips_in_minute.duration /= 60

trips_in_minute.head()
trips_in_minute.isnull().sum()
trips_in_minute.describe()
trips_in_minute.plot(x='start_date' , y = 'duration', style='o',figsize=(15,10))



plt.show()
trips_in_minute = trips_in_minute[trips_in_minute.duration <= 370]

trips_in_minute = trips_in_minute[trips_in_minute.duration >3]



trips_in_minute.plot(x='start_date' , y = 'duration', style='o',figsize=(20,10))



plt.show()
trips_weekday = trips_in_minute

trips_weekday['weekday'] =  pd.to_datetime(trips_weekday['start_date']).dt.weekday_name

trips_weekday.groupby('weekday').count()['start_date'].sort_values(ascending=False)[0:19].plot(kind='bar',figsize=(14,4));
trips_weekday.groupby('weekday').mean()['duration'].sort_values(ascending=False)[0:7].plot(kind='bar',figsize=(14,4));
trips_month = trips_in_minute

trips_month['month'] =  pd.to_datetime(trips_month['start_date']).dt.month



trips_month.groupby('month').count()['start_date'][0:12].plot(kind='bar',figsize=(13,5));
weathers = pd.read_csv("../input/weather.csv")

weathers['month'] =  pd.to_datetime(weathers['date']).dt.month

weathers.head()
weathers.groupby('month').mean()['mean_temperature_f'][0:12].plot(kind='line',figsize=(10,5));
# Convierto el campo date en datetime

weather_convert = weathers

weather_convert.date = pd.to_datetime(weather_convert.date)

weather_convert.precipitation_inches = pd.to_numeric(weather_convert.precipitation_inches, errors='coerce')

weather_convert = weather_convert[np.isfinite(weather_convert['precipitation_inches'])]

weather_convert.head()
weather_convert.describe()
weather_convert.info()
# Compruebo que no se repitan las fechas

nUniqueDate = weather_convert.date.nunique()

print ("Fechas distintas:")

print (nUniqueDate)

nUniqueZipCode = weather_convert.zip_code.nunique()

print ("Zip codes distintos:")

print (nUniqueZipCode)



print (nUniqueDate*nUniqueZipCode)
print(weather_convert.groupby('precipitation_inches').size())
weather_convert['month'] =  pd.to_datetime(weather_convert['date']).dt.month



grouped_weather = weather_convert.groupby('month')['precipitation_inches'].sum()



grouped_weather.plot(title="Suma de las precipitaciones totales por mes \n(acumulado desde el 29/8/2013 hasta el 31/8/2015)", xticks=[1,2,3,4,5,6,7,8,9,10,11,12])

plt.xlabel("mes")

plt.ylabel("precipitaciones [pulgadas]")
print (weather_convert.events.unique())

print (weather_convert.zip_code.unique())
# Se pasa mean_temperature a grados celsius

weather_convert['mean_temperature_f'] = (weather_convert.mean_temperature_f - 32)/1.8
cityZipCodes = pd.DataFrame(data=[['San Francisco',94107],['San Jose',95113],['Redwood City',94063],['Mountain View',94041],['Palo Alto',94301]], columns=['city', 'zip_code'])

print (cityZipCodes)
trip_convert = trips



trip_convert.zip_code = pd.to_numeric(trip_convert.zip_code, errors='coerce')

trip_convert = trip_convert[np.isfinite(trip_convert['zip_code'])]



trip_convert['dia'] = trip_convert.start_date.dt.weekday_name



trip_convert['start_date'] = trip_convert.start_date.dt.normalize()
station_filter = stations.loc[:,['id','city']]

station_filter
dfMergedZipCodes = pd.merge(weather_convert, cityZipCodes, how='inner', on='zip_code')

dfMergedZipCodes
tripsMergedCity = pd.merge(trip_convert, station_filter, how='inner', left_on='start_station_id', right_on='id')

tripsMergedCity
tripsMonday = tripsMergedCity[tripsMergedCity.dia == 'Monday']

tripsTuesday = tripsMergedCity[tripsMergedCity.dia == 'Tuesday']

tripsWednesday = tripsMergedCity[tripsMergedCity.dia == 'Wednesday']

tripsThursday = tripsMergedCity[tripsMergedCity.dia == 'Thursday']

tripsFriday = tripsMergedCity[tripsMergedCity.dia == 'Friday']

tripsSaturday = tripsMergedCity[tripsMergedCity.dia == 'Saturday']

tripsSunday = tripsMergedCity[tripsMergedCity.dia == 'Sunday']
print (trip_convert.info())
dfMergedMonday = pd.merge(tripsMonday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedThursday = pd.merge(tripsThursday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedWednesday = pd.merge(tripsWednesday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedTuesday = pd.merge(tripsTuesday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedFriday = pd.merge(tripsFriday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedSaturday = pd.merge(tripsSaturday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedSunday = pd.merge(tripsSunday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])
dfMergedMonday = pd.merge(tripsMonday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedThursday = pd.merge(tripsThursday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedWednesday = pd.merge(tripsWednesday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedTuesday = pd.merge(tripsTuesday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedFriday = pd.merge(tripsFriday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedSaturday = pd.merge(tripsSaturday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])

dfMergedSunday = pd.merge(tripsSunday, dfMergedZipCodes, left_on=['city','start_date'], right_on=['city','date'])
dfMergedGroupedCountMonday = dfMergedMonday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountTuesday = dfMergedTuesday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountWednesday = dfMergedWednesday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountThursday = dfMergedThursday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountFriday = dfMergedFriday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountSaturday = dfMergedSaturday.groupby('mean_temperature_f', as_index=False)['duration'].count()

dfMergedGroupedCountSunday = dfMergedSunday.groupby('mean_temperature_f', as_index=False)['duration'].count()
dfMergedGroupedCountMonday
dfMergedGroupedMeanMonday = dfMergedMonday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanMonday.columns = ['mean_temperature_f', 'Lunes']

dfMergedGroupedMeanTuesday = dfMergedTuesday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanTuesday.columns = ['mean_temperature_f', 'Martes']

dfMergedGroupedMeanWednesday = dfMergedWednesday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanWednesday.columns = ['mean_temperature_f', 'Miercoles']

dfMergedGroupedMeanThursday = dfMergedThursday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanThursday.columns = ['mean_temperature_f', 'Jueves']

dfMergedGroupedMeanFriday = dfMergedFriday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanFriday.columns = ['mean_temperature_f', 'Viernes']

dfMergedGroupedMeanSaturday = dfMergedSaturday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanSaturday.columns = ['mean_temperature_f', 'Sabado']

dfMergedGroupedMeanSunday = dfMergedSunday.groupby('mean_temperature_f', as_index=False)['duration'].mean()

dfMergedGroupedMeanSunday.columns = ['mean_temperature_f', 'Domingo']

#dfMergedGroupedMean



dfMergedGroupedMean = pd.merge(dfMergedGroupedMeanMonday, dfMergedGroupedMeanTuesday, on='mean_temperature_f')

dfMergedGroupedMean = pd.merge(dfMergedGroupedMean, dfMergedGroupedMeanWednesday, on='mean_temperature_f')

dfMergedGroupedMean = pd.merge(dfMergedGroupedMean, dfMergedGroupedMeanThursday, on='mean_temperature_f')

dfMergedGroupedMean = pd.merge(dfMergedGroupedMean, dfMergedGroupedMeanFriday, on='mean_temperature_f')

dfMergedGroupedMean = pd.merge(dfMergedGroupedMean, dfMergedGroupedMeanSaturday, on='mean_temperature_f')

dfMergedGroupedMean = pd.merge(dfMergedGroupedMean, dfMergedGroupedMeanSunday, on='mean_temperature_f')

dfMergedGroupedMean
dfPlot = dfMergedGroupedMean[(dfMergedGroupedMean.mean_temperature_f >= 11.66666) & (dfMergedGroupedMean.mean_temperature_f <= 21.2)]

#plot(x='mean_temperature_f',y='duration')





dfPlot.plot(x='mean_temperature_f', title="Promedio de la duracion de los viajes segun temperatura media del dia",

            color=['#3336FF','#33FF49','#FF3333','#F3FF33','#55F1F1','#000000','#93881A'])

plt.legend = [1,2,3,4,5]

plt.xlabel("Temperatura [grad. celsius]")

plt.ylabel("Duracion [seg]")

#plt.ylim([0,9000])

#plt.colors = ['b','o','r','y','m']

#plt.figure()

#with pd.plot_params.use('x_compat', True):

 #   dfPlot.duration.plot(color='r')
dfPlot
dfPlot.info()
grupo_trip = trips_in_minute.loc[:,['start_station_id','duration']]



grupo_trip['subindex'] = grupo_trip.groupby('start_station_id').cumcount() + 1

grupo_trip = pd.pivot_table(grupo_trip,index='subindex',columns='start_station_id',values='duration')



grupo_trip.describe()
grupo_trip.plot.box(figsize=(15,5));
trips_in_minute.plot.scatter('end_station_id','start_station_id',alpha=1,title = "Relacion entre estaciones",figsize=(15,5));
viajesInterCiudad = trips_in_minute[trips_in_minute.end_station_id == 2]

viajesInterCiudad = viajesInterCiudad[viajesInterCiudad.start_station_id > 60]

viajesInterCiudad.head()
plt.figure()

plt.ylabel = "1"

trips_in_minute.groupby('start_station_name').count()['id'].sort_values(ascending=False)[0:30].plot(title = "Estaciones de donde salen mas viajes",kind='bar',figsize=(14,4))
trips_in_minute.groupby('end_station_name').count()['id'].sort_values(ascending=False)[0:30].plot(kind='bar',figsize=(14,5))
'''gmap = gmplot.GoogleMapPlotter(37.336, -121.894074, 16)

gmap.heatmap(stations.lat, stations.long)  

gmap.draw("concetraciónDeEstaciones.html")'''
'''googlePlotMap= trips_in_minute.loc[:,['start_station_id','end_station_id']]

googlePlotMap.columns = ['id' , 'end_station_id']

stationPlot = stations.loc[:,['id','lat','long']]

innerJoin = pd.merge(googlePlotMap,stationPlot, on='id', how='inner')





gmap = gmplot.GoogleMapPlotter(37.336, -121.894074, 16)

gmap.heatmap(innerJoin.lat, innerJoin.long)  

gmap.draw("startStationIdHeatMap.html")'''
#Heatmap a donde llegan la mayor cantidad de viajes. 

'''googlePlotMap= trips_in_minute.loc[:,['start_station_id','end_station_id']]

googlePlotMap.columns = ['start_station_id' , 'id']

stationPlot = stations.loc[:,['id','lat','long']]

innerJoin = pd.merge(googlePlotMap,stationPlot, on='id', how='inner')



gmap = gmplot.GoogleMapPlotter(37.336, -121.894074, 16)

gmap.heatmap(innerJoin.lat, innerJoin.long)  

gmap.draw("endStationIdHeatMap.html")'''
'''googlePlotMap= trips_in_minute.loc[:,['start_station_id','end_station_id']]

googlePlotMap.columns = ['id' , 'end_station_id']





groupedby = googlePlotMap.groupby('id',as_index=False).count()

stationPlot = stations.loc[:,['id','lat','long']]

innerJoin = pd.merge(groupedby,stationPlot, on='id', how='inner')



import gmplot



gmap = gmplot.GoogleMapPlotter(37.336, -121.894074, 16)

#gmap.plot(innerJoin.lat, innerJoin.long, 'cornflowerblue', edge_width=4)

for index, row in innerJoin.iterrows():

    gmap.scatter([row['lat']], [row['long']], '#FFFFFF', row['end_station_id']*0.005, marker=False)

    gmap.marker(row['lat'],row['long'],color='#FF0000',title=str(row['id']))

    

gmap.draw("mymap.html")'''
'''googlePlotMap= trips_in_minute.loc[:,['start_station_id','end_station_id']]

googlePlotMap.columns = ['id' , 'end_station_id']



groupedby = googlePlotMap.groupby('id',as_index=False).count()

stationPlot = stations.loc[:,['id','lat','long']]

#stationPlot = stationPlot[stationPlot.lat > 38]

#stationPlot = stationPlot[stationPlot.long < -121]



innerJoin = pd.merge(groupedby,stationPlot, on='id', how='inner')

innerJoin =  innerJoin[innerJoin.id > 38] 

innerJoin =  innerJoin[innerJoin.id < 80] 



import gmplot

gmap.from_geocode("San Francisco")

# gmap = gmplot.GoogleMapPlotter(3.336, -121.894074, 16)

gmap.heatmap(innerJoin.lat, innerJoin.long)  



# #gmap.plot(innerJoin.lat, innerJoin.long, 'cornflowerblue', edge_width=4)

# for index, row in innerJoin.iterrows():

#     gmap.scatter([row['lat']], [row['long']], '#FFFFFF', row['end_station_id']*0.005, marker=False)

#     gmap.marker(row['lat'],row['long'],color='#FF0000',title=str(row['id']))

    

gmap.draw("mymap.html")'''
from_to = trips_in_minute[trips_in_minute['end_station_id'] == 2].groupby('start_station_id').size()
fig, ax = plt.subplots(figsize=(18,12))        



sns.heatmap(pd.crosstab(trips_in_minute['start_station_id'],trips_in_minute['end_station_id']));
'''status = pd.read_csv("../input/status.csv")'''
# Convierto el campo time en datetime

'''status_convert = status

status_convert.time = pd.to_datetime(status_convert.time) 



print status_convert'''
'''status_convert.info(null_counts=True)'''
'''status_convert.describe()'''
# Agrego el campo weekday y hour, y elimino time para liberar un poco la memoria

'''status_convert['weekday'] =  pd.to_datetime(status_convert['time']).dt.weekday

status_convert['hour'] = pd.to_datetime(status_convert['time']).dt.hour

status_convert = status_convert.drop('time', 1)'''
# Esto va a servir mas adelante para graficar

'''days = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']

labels = []

for day in days:

    for h in range(12):

        labels.append(day + ' - ' + str(h*2))'''
# Estacion 70

'''df_station = status_convert[status_convert.station_id == 70]

print (df_station)'''

#dfGrouped = df.groupby(['weekday',df.hour-(df.hour%2)])['bikes_available'].count()

#print dfGrouped
'''dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

print dfGrouped'''
'''plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora en la estacion San Francisco Caltrain (Townsend at 4th)").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
# Estación 69

'''df_station = status_convert[status_convert.station_id == 69]

dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora San Francisco Caltrain 2 (330 Townsend)").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
# Estación 28

'''df_station = status_convert[status_convert.station_id == 28]

dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora Mountain View Caltrain Station").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
# Estación 46

'''df_station = status_convert[status_convert.station_id == 46]

dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora Washington at Kearney").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
# Estación 21

'''df_station = status_convert[status_convert.station_id == 21]

dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora Franklin at Maple").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
# Estación 24

'''df_station = status_convert[status_convert.station_id == 24]

dfGrouped = df_station.groupby(['weekday',df_station.hour-(df_station.hour%2)])['bikes_available'].mean()

plt.figure()

dfGrouped.plot(kind='bar',figsize=(30,8), title="Disponibilidad de bicicletas segun dia y hora Redwood City Public Library").set_xticklabels(labels)

plt.xlabel("dia - hora")'''

#plt.ylabel("disponibilidad de bicicletas")
bikes = trips_in_minute.loc[:,'bike_id']

bikes
print ((trips_in_minute.bike_id.unique()).min())

print ((trips_in_minute.bike_id.unique()).max()) 

print (len(trips_in_minute.bike_id.unique())) 
#%matplotlib notebook

bikes.plot.hist(alpha=0.5, title="Cantidad de viajes por bicicleta")

#plt.xlabel("bike_id")

#plt.ylabel("cantidad de viajes")


trips_in_minute.plot.scatter('start_station_id','bike_id',alpha=1,title = "Relacion bicicletas-estaciones de salida",figsize=(15,8));


trips_in_minute.plot.scatter('end_station_id','bike_id',alpha=1,title = "Relacion bicicletas-estaciones de llegada",figsize=(15,8));