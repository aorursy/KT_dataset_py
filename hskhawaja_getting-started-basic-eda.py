import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(12,5)

df = pd.read_csv("../input/flights.csv")

# Taking a subset to save computational time

df = df[df['MONTH'] == 1]

df.head()
airlineList = df['AIRLINE'].unique()

airlineList = airlineList.tolist()
def calculate_Airline_D_Delays(airlineName):

    d = df[df['AIRLINE'] == airlineName]

    d = d[d['DEPARTURE_DELAY'] > 0]

    li = d['DEPARTURE_DELAY'].tolist()

    li = np.array(li)

    return li



def calculate_Airline_A_Delays(airlineName):

    d = df[df['AIRLINE'] == airlineName]

    d = d[d['ARRIVAL_DELAY'] > 0]

    li = d['ARRIVAL_DELAY'].tolist()

    li = np.array(li)

    return li
avgAirlineDD = []

avgAirlineAD = []

for a in airlineList:

    avgAirlineDD.append(calculate_Airline_D_Delays(a).mean())

    avgAirlineAD.append(calculate_Airline_A_Delays(a).mean())
n_groups = len(airlineList)



fig, ax = plt.subplots()



index = np.arange(n_groups)

bar_width = 0.25



opacity = 0.4

error_config = {'ecolor': '0.3'}



rects1 = plt.bar(index, avgAirlineDD, bar_width,

                 alpha=opacity,

                 color='b',

                 error_kw=error_config,

                 label='Departure')



rects2 = plt.bar(index + bar_width, avgAirlineAD, bar_width,

                 alpha=opacity,

                 color='r',

                 error_kw=error_config,

                 label='Arrival')



plt.margins(0.01)



plt.xlabel('Airlines')

plt.ylabel('Average Delays (Min)')

plt.title('Comparison of Departure/Arrival Delays')

plt.xticks(index + bar_width / 2, airlineList)

plt.legend(loc = 'upper left')



plt.tight_layout()

plt.show()
def calculate_Airport_D_Delays(airportName):

    d = df[df['ORIGIN_AIRPORT'] == airportName]

    d = d[d['DEPARTURE_DELAY'] > 0]

    li = d['DEPARTURE_DELAY'].tolist()

    li = np.array(li)

    return li



def calculate_Airport_A_Delays(airportName):

    d = df[df['DESTINATION_AIRPORT'] == airportName]

    d = d[d['ARRIVAL_DELAY'] > 0]

    li = d['ARRIVAL_DELAY'].tolist()

    li = np.array(li)

    return li
airportDepList = df['ORIGIN_AIRPORT'].unique()

airportDepList = airportDepList.tolist()

airportArrList = df['DESTINATION_AIRPORT'].unique()

airportArrList = airportArrList.tolist()
avgAirportDD = []

avgAirportAD = []

for a in airportDepList:

    avgAirportDD.append(calculate_Airport_D_Delays(a).mean())

for a in airportArrList:

    avgAirportAD.append(calculate_Airport_A_Delays(a).mean())
x = zip(airportDepList, avgAirportDD)

x = sorted(x, key=lambda item:item[1])

names = []

values = []

x = x[-20:]

for i,j in x:

    names.append(i)

    values.append(j)
n_groups = len(names)



index = np.arange(n_groups)

bar_width = 0.6



opacity = 0.4

error_config = {'ecolor': '0.3'}



rects1 = plt.bar(index, values, bar_width,

                 alpha=opacity,

                 color='b',

                 error_kw=error_config,

                 label='Departure')



plt.margins(0.01)



plt.xlabel('Airports')

plt.ylabel('Average Delays (Min)')

plt.title('Top 20 Airports with most Departure Delays')

plt.xticks(index + bar_width / 2, names)

plt.legend(loc = 'upper left')



plt.tight_layout()

plt.show()