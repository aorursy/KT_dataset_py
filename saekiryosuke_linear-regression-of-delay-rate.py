import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# import csv



df = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')



print(df.head())
boolean15MinutesDelay = (df['ARR_DEL15']==1.0) & (df['DEP_DEL15']==0.0)



dayOfWeek_onlydelay = df.DAY_OF_WEEK[boolean15MinutesDelay]



labels = ['all', 'arrive delay']

fig = plt.figure()

ax = fig.add_subplot(111, xlabel='day in a week', ylabel='number of flights')

fig = plt.hist([df.DAY_OF_WEEK, dayOfWeek_onlydelay], label=labels, bins = 7)

fig = plt.legend()

fig = plt.title("number fo flight vs day in a week")



# histogram of only delay cases

fig = plt.figure()

ax = fig.add_subplot(111, xlabel='day in a week', ylabel='number of flights')

fig = plt.hist(dayOfWeek_onlydelay, label=labels, bins=7)

fig = plt.title("number fo flight vs day in a week")
departureTime_onlydelay = df.ARR_TIME[boolean15MinutesDelay]



labels = ['all', 'delay']



fig = plt.figure()

ax = fig.add_subplot(111, xlabel='24 hour time', ylabel='number of flights')



fig = plt.hist([df.ARR_TIME, departureTime_onlydelay], label=labels, bins=12)

fig = plt.legend()

fig = plt.title("number fo flight vs 24 hour time")



# Time is expressed as hhmm (ex. "2000" means "20:00").
# histogram of only delay cases.

fig = plt.figure()

ax = fig.add_subplot(111, xlabel='24 hour time', ylabel='number of flights')



fig = plt.hist(departureTime_onlydelay, bins=12)

fig = plt.title("number fo flight vs 24 hour time")

carrierID_onlydelay = df.OP_CARRIER_AIRLINE_ID[boolean15MinutesDelay]



# all flights

airline_IDs = df['OP_CARRIER_AIRLINE_ID'].value_counts()



# number of delay

delay_airline_IDs = carrierID_onlydelay.value_counts()



df_allflights = pd.DataFrame({'airlineID': airline_IDs.index, 'flights': airline_IDs.values})



# make series delay_airline_IDs -> dataframe

df_delayflights = pd.DataFrame({'airlineID': delay_airline_IDs.index, 'arrive_delay_flights': delay_airline_IDs.values})



# combine dataframe df_allflights, df_delayflights

stats = pd.merge(df_allflights, df_delayflights)



# delay rate[%]

delayRate = stats['arrive_delay_flights'] / stats['flights'] * 100

df_delay_rate = pd.DataFrame({'airlineID': delay_airline_IDs.index, 'arrive_delay_rate(%)': delayRate})



stats = pd.merge(stats, df_delay_rate)



print(stats)

stats.plot(x='airlineID', y=['flights', 'arrive_delay_rate(%)'], kind='bar', subplots=True)

flight_distance_onlydelay = df.DISTANCE[boolean15MinutesDelay]



labels = ['all', 'delay']

fig = plt.figure()

ax = fig.add_subplot(111, xlabel='mile', ylabel='number of flights')



fig = histDelayrateVsDistance = plt.hist([df.DISTANCE, flight_distance_onlydelay], label=labels, bins=50)

fig = plt.legend()
histDelayrateVsDistance = list(histDelayrateVsDistance)



i = 0



rate = []

while i <= 49:

    rate.append((histDelayrateVsDistance[0][1][i] / (histDelayrateVsDistance[0][0][i] + 0.0001))*100)

    i = i + 1



xAxis = histDelayrateVsDistance[1]



fig = plt.figure()

ax = fig.add_subplot(111, xlabel='mile', ylabel='delay rate(%)')



fig = plt.plot(xAxis[1:], rate)



booleanUnder2700mile = (df['DISTANCE']<=2700.0)

distanceUnder2700mile = df.DISTANCE[booleanUnder2700mile]



print('How much(%) under 2700 mile flights occupy?')

print(len(distanceUnder2700mile) / len(df['DISTANCE']) * 100)
# I tried linear fitting.

a, b = np.polyfit(xAxis[1:29], rate[0:28], 1)

print(a, b)



# y = ax + b

# y[%] = 1.08e-3 * x[mile] + 3.62

fitResult = np.array(xAxis[1:29]) * a + b



fig = plt.figure()

ax = fig.add_subplot(111, xlabel='mile', ylabel='rate(%)')



fig = plt.plot(xAxis[1:28], rate[1:28])

fig = plt.plot(xAxis[1:28], fitResult[1:28])

plt.text(100, 6, 'fittin result : $y=1.08e-3 * x + 3.62$')