import pandas as pd

import numpy as np

import matplotlib.pylab as plt

from pandas.plotting import radviz

import matplotlib

matplotlib.style.use('ggplot')

from subprocess import check_output



%matplotlib inline
data = pd.read_csv("../input/flight-delays/flights.csv")

data.head(100)
Total_delay = data['ARRIVAL_DELAY'].sum()

Total_rows = len(data.index)



print(Total_delay / Total_rows)
'''



NAS Delay:



Delay that is within the control of the National Airspace System (NAS) may include: non-extreme weather

conditions, airport operations, heavy traffic volume, air traffic control, etc. Delays that occur after 

Actual Gate Out are usually attributed to the NAS and are also reported through OPSNET.







OPSNET Delay Cause:



Delays to Instrument Flight Rules (IFR) traffic of 15 minutes or more, experienced by individual flights, 

which result from the ATC system detaining an aircraft at the gate, short of the runway, on the runway, 

on a taxiway, and/or in a holding configuration anywhere en route.



Such delays include delays due to weather conditions at airports and en route (Weather), FAA and non-FAA 

equipment malfunctions (Equipment), the volume of traffic at an airport (Volume), reduction to runway capacity 

(Runway), and other factors (Others). Flight delays of less than 15 minutes are not reported in OPSNET. 

ASPM reports the most dominant OPSNET delay cause for any flight with an ASQP Reported NAS Delay.



'''



df = data.loc[(data['AIR_SYSTEM_DELAY'] != False, ['ARRIVAL_DELAY'])]



delays_airsys = data.loc[(data['AIR_SYSTEM_DELAY'] > 0, ['ARRIVAL_DELAY'])].sum()

notdelayed_airsys = data.loc[(data['AIR_SYSTEM_DELAY'] <= 0, ['ARRIVAL_DELAY'])].sum()



count_sys = df[(data['AIR_SYSTEM_DELAY'] != False)].count()

count_sys_pos = data.loc[(data['AIR_SYSTEM_DELAY'] > 0, ['ARRIVAL_DELAY'])].count()





# -------------------------------------------------------------------------------------------------------------



'''



Security Delay:



Security delay is caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security 

breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.



'''





df = data.loc[(data['SECURITY_DELAY'] != False, ['ARRIVAL_DELAY'])]



delays_airsec = data.loc[(data['SECURITY_DELAY'] > 0, ['ARRIVAL_DELAY'])].sum()

notdelayed_airsec = data.loc[(data['SECURITY_DELAY'] <= 0, ['ARRIVAL_DELAY'])].sum()



count_sec = df[(data['SECURITY_DELAY'] != False)].count()

count_sec_pos = data.loc[(data['SECURITY_DELAY'] > 0, ['ARRIVAL_DELAY'])].count()





# -------------------------------------------------------------------------------------------------------------



'''



Carrier Delay:



Carrier delay is within the control of the air carrier. Examples of occurrences that may determine carrier delay

are: aircraft cleaning, aircraft damage, awaiting the arrival of connecting passengers or crew, baggage, bird 

strike, cargo loading, catering, computer, outage-carrier equipment, crew legality (pilot or attendant rest), 

damage by hazardous goods, engineering inspection, fueling, handling disabled passengers, late crew, lavatory 

servicing, maintenance, oversales, potable water servicing, removal of unruly passenger, slow boarding or seating, 

stowing carry-on baggage, weight and balance delays.



'''



df = data.loc[(data['AIRLINE_DELAY'] != False, ['ARRIVAL_DELAY'])]



delays_airair = data.loc[(data['AIRLINE_DELAY'] > 0, ['ARRIVAL_DELAY'])].sum()

notdelayed_airair = data.loc[(data['AIRLINE_DELAY'] <= 0, ['ARRIVAL_DELAY'])].sum()



count_air = df[(data['AIRLINE_DELAY'] != False)].count()

count_air_pos = data.loc[(data['AIRLINE_DELAY'] > 0, ['ARRIVAL_DELAY'])].count()





# -------------------------------------------------------------------------------------------------------------



'''



Late Arrival Delay:



Arrival delay at an airport due to the late arrival of the same aircraft at a previous airport. The ripple 

effect of an earlier delay at downstream airports is referred to as delay propagation.



'''



df = data.loc[(data['LATE_AIRCRAFT_DELAY'] != False, ['ARRIVAL_DELAY'])]



delays_airlate = data.loc[(data['LATE_AIRCRAFT_DELAY'] > 0, ['ARRIVAL_DELAY'])].sum()

notdelayed_airlate = data.loc[(data['LATE_AIRCRAFT_DELAY'] <= 0, ['ARRIVAL_DELAY'])].sum()



count_late = df[(data['LATE_AIRCRAFT_DELAY'] != False)].count()

count_late_pos = data.loc[(data['LATE_AIRCRAFT_DELAY'] > 0, ['ARRIVAL_DELAY'])].count()





# -------------------------------------------------------------------------------------------------------------



'''



Weather Delay:



Weather delay is caused by extreme or hazardous weather conditions that are forecasted or manifest themselves

on point of departure, enroute, or on point of arrival.



'''



df = data.loc[(data['WEATHER_DELAY'] != False, ['ARRIVAL_DELAY'])]



delays_airweather = data.loc[(data['WEATHER_DELAY'] > 0, ['ARRIVAL_DELAY'])].sum()

notdelayed_airweather = data.loc[(data['WEATHER_DELAY'] <= 0, ['ARRIVAL_DELAY'])].sum()



count_weather = df[(data['WEATHER_DELAY'] != False)].count()

count_weather_pos = data.loc[(data['WEATHER_DELAY'] > 0, ['ARRIVAL_DELAY'])].count()





# -------------------------------------------------------------------------------------------------------------





delays = []

only_delays = []



# Average delay time accounting for delayed and not delayed flights



system = (delays_airsys + notdelayed_airsys) / count_sys

security = (delays_airsec + notdelayed_airsec) / count_sec

airline = (delays_airair + notdelayed_airair) / count_air

late = (delays_airlate + notdelayed_airlate) / count_late

weather = (delays_airweather + notdelayed_airweather) / count_weather



# Average delay time NOT accounting non-delayed flights



system_pos = delays_airsys / count_sys_pos

security_pos = delays_airsec / count_sec_pos

airline_pos = delays_airair / count_air_pos

late_pos = delays_airlate / count_late_pos

weather_pos = delays_airweather / count_weather_pos



# Append to array to ease access later



delays.append(system)

delays.append(security)

delays.append(airline)

delays.append(late)

delays.append(weather)



# Append only delayed flights



only_delays.append(system_pos)

only_delays.append(security_pos)

only_delays.append(airline_pos)

only_delays.append(late_pos)

only_delays.append(weather_pos)



# Print average times



print(delays)

print(only_delays)
# Fix figure size



plt.figure(figsize=(20, 8))





# Print each delayed flight due to all reasons



ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='AIR_SYSTEM_DELAY',figsize=(20,8), color='r')

ax2 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='SECURITY_DELAY', color='b', ax=ax1)

ax3 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='AIRLINE_DELAY', color='y', ax=ax1)

ax4 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='LATE_AIRCRAFT_DELAY', color='g', ax=ax1)

ax5 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='WEATHER_DELAY', color='m', ax=ax1)





# Plot, plot, plot!



ax1.set_xlabel("Time (HHMM)")

ax1.set_ylabel("Time Delayed")

plt.title('All Delays')
plt.figure(figsize=(20, 8))

ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='AIR_SYSTEM_DELAY', figsize=(20,8), color='r')

ax1.set_xlabel("Time (HHMM)")

ax1.set_ylabel("Time Delayed")

plt.title('Air System Delays')
plt.figure(figsize=(20, 8))

ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='SECURITY_DELAY', figsize=(20,8), color='b')

ax1.set_xlabel("Time")

ax1.set_ylabel("Time Delayed")

plt.title('Security Delays')
plt.figure(figsize=(20, 8))

ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='AIRLINE_DELAY', figsize=(20,8), color='y')

ax1.set_xlabel("Time")

ax1.set_ylabel("Time Delayed")

plt.title('Carrier Delays')
plt.figure(figsize=(20, 8))

ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='LATE_AIRCRAFT_DELAY', figsize=(20,8), color='g')

ax1.set_xlabel("Time")

ax1.set_ylabel("Time Delayed")

plt.title('Late Aircraft Arrival Delays')
plt.figure(figsize=(20, 8))

ax1 = data.plot(kind='scatter', x='SCHEDULED_DEPARTURE', y='WEATHER_DELAY', figsize=(20,8), color='m')

ax1.set_xlabel("Time")

ax1.set_ylabel("Time Delayed")

plt.title('Weather Delays')
objects = ('System', 'Security', 'Airline', 'Late Arrival', 'Weather')

y_pos = np.arange(len(objects))

performance = [system,security,airline,late,weather]

 

plt.figure(figsize=(16,6))

plt.bar(y_pos, performance, align='center', alpha=0.75, color='black')

plt.xticks(y_pos, objects)

plt.ylabel('Average Length')

plt.title('Delay Types')



plt.show()
objects = ('System', 'Security', 'Airline', 'Late Arrival', 'Weather')

y_pos = np.arange(len(objects))

performance = [system_pos,security_pos,airline_pos,late_pos,weather_pos]

 

plt.figure(figsize=(16,6))

plt.bar(y_pos, performance, align='center', alpha=0.75, color='purple')

plt.xticks(y_pos, objects)

plt.ylabel('Average Length (minutes)')

plt.xlabel('Delay Type')

plt.title('Average Delay Time v. Delay Type')



plt.show()
labels = 'Weather', 'Late Arrival', 'Carrier', 'Security', 'Air System'

sizes = [delays_airweather, delays_airlate, delays_airair, delays_airsec, delays_airsys]

explode = (0.1, 0.1, 0.1, 0.1, 0.1)

colors=['magenta','green','yellow','blue','red']



fig1, ax1 = plt.subplots(figsize=(8,8))

ax1.pie(sizes, colors=colors,labels=labels, autopct='%1.2f%%',

        shadow=True, explode=explode, startangle=90, center=(0,0))

ax1.axis('equal')





plt.show()