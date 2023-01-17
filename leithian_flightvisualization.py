import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



dt = np.dtype([("YEAR", "int64"),                 ("MONTH", "int64"),              ("DAY", "int64"),              ("DAY_OF_WEEK", "int64"),          ("AIRLINE", "object"),

               ("FLIGHT_NUMBER", "int64"),        ("TAIL_NUMBER", "object"),       ("ORIGIN_AIRPORT", "object"),  ("DESTINATION_AIRPORT", "object"), ("SCHEDULED_DEPARTURE", "int64"),

               ("DEPARTURE_TIME", "float64"),     ("DEPARTURE_DELAY", "float64"),  ("TAXI_OUT", "float64"),       ("WHEELS_OFF", "float64"),         ("SCHEDULED_TIME", "float64"),

               ("ELAPSED_TIME", "float64"),       ("AIR_TIME", "float64"),         ("DISTANCE", "int64"),         ("WHEELS_ON", "float64"),          ("TAXI_IN", "float64"),

               ("SCHEDULED_ARRIVAL", "int64"),    ("ARRIVAL_TIME", "float64"),     ("ARRIVAL_DELAY", "float64"),  ("DIVERTED", "int64"),             ("CANCELLED", "int64"),

               ("CANCELLATION_REASON", "object"), ("AIR_SYSTEM_DELAY", "float64"), ("SECURITY_DELAY", "float64"), ("AIRLINE_DELAY", "float64"),      ("LATE_AIRCRAFT_DELAY", "float64"),

               ("WEATHER_DELAY", "float64")])



# Any results you write to the current directory are saved as output.

airlines = pd.read_csv('../input/flight-delays/airlines.csv')

flights = pd.read_csv('../input/flight-delays/flights.csv', dtype=dt)

airports = pd.read_csv('../input/flight-delays/airports.csv')

flightNumbersArr = np.load('../input/flightnumberscounted/flightNumbersArr.npy')

flightNumbersDep = np.load('../input/flightnumberscounted/flightNumbersDep.npy')

airports["numbersArr"] = flightNumbersArr

airports["numbersDep"] = flightNumbersDep
size = np.full(np.shape(len(airports)), 99)

m = Basemap(llcrnrlon=-180, llcrnrlat=10, urcrnrlon=-60, urcrnrlat=75)

plt.figure(figsize=(20,15))

m.bluemarble()

m.scatter(airports["LONGITUDE"], airports["LATITUDE"], c=airports["numbersArr"], s=size, cmap=plt.cm.viridis, norm=matplotlib.colors.LogNorm(), vmin=1)

m.colorbar()

#plt.savefig('../output/AirportsByArr.pdf')
m = Basemap(llcrnrlon=-180, llcrnrlat=10, urcrnrlon=-60, urcrnrlat=75)

plt.figure(figsize=(20,13))

m.bluemarble()

m.scatter(airports["LONGITUDE"], airports["LATITUDE"], c=airports["numbersDep"], s=size, cmap=plt.cm.viridis, norm=matplotlib.colors.LogNorm(), vmin=1)

m.colorbar()

#plt.savefig('../AirportsByDep.pdf')
np.sum(flights["CANCELLED"])
np.mean(flights.TAXI_OUT)
np.std(flights.TAXI_OUT)
np.mean(np.abs(flights.DEPARTURE_DELAY - np.full(fill_value=np.mean(flights.DEPARTURE_DELAY), shape=flights.DEPARTURE_DELAY.shape[0])))
np.std(np.abs(flights.DEPARTURE_DELAY - np.full(fill_value=np.mean(flights.DEPARTURE_DELAY), shape=flights.DEPARTURE_DELAY.shape[0])))