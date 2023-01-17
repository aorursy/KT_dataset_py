#Airlines Analysis
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
airlines = pd.read_csv("../input/airlines.csv")

airports = pd.read_csv("../input/airports.csv")
flights = pd.read_csv("../input/flights.csv", low_memory = False)

valid_airports = []

for code in airports['IATA_CODE']:

    valid_airports.append(code)

flights = flights[flights['ORIGIN_AIRPORT'].isin(valid_airports)]

flights = flights[flights['DESTINATION_AIRPORT'].isin(valid_airports)]

print ("Some airports in Flights Dataframe were not present in the Airport Dataframe, and hence were entered incorrectly.")

print ("All those rows were dropped.")
print ("The general structure of the Datasets-")

print (airlines.columns)

print (airlines.head(2))



print (airports.columns)

print (airports.head(2))



print (flights.columns)

print (flights.head(2))
airline_dictionary = dict(zip(airlines['IATA_CODE'], airlines['AIRLINE']))

print ("A dictionary of airline Codes and Names for easy reference.")

print (airline_dictionary)
delays = flights[['AIRLINE', 'ORIGIN_AIRPORT', 

                  'DESTINATION_AIRPORT', 'MONTH', 

                  'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 

                  'SECURITY_DELAY', 'DEPARTURE_DELAY',

                  'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY']]

delays = delays.fillna(0)

delays['TOTAL_DELAY_AIRLINE'] = delays['AIRLINE_DELAY']+ delays['LATE_AIRCRAFT_DELAY']

delays['TOTAL_DELAY_SOURCE_AIRPORT'] = delays['SECURITY_DELAY'] + delays['DEPARTURE_DELAY'] + delays['AIR_SYSTEM_DELAY']

delays['TOTAL_DELAY_DEST_AIRPORT'] = delays['ARRIVAL_DELAY'] + delays['AIR_SYSTEM_DELAY']
monthly_delays = delays[['MONTH', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 

                         'SECURITY_DELAY', 'DEPARTURE_DELAY', 

                         'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY']].groupby('MONTH').mean().reset_index()



p1 = plt.figure(figsize = (10, 10))

dim = p1.add_axes([0,0,1,1])

for delay_type in monthly_delays:

    if delay_type is not 'MONTH':

        monthly_delays[delay_type].plot.line(x = 'MONTH', label = delay_type)

dim.legend();

plt.title('Trends in Delays over the months.')

plt.xlabel('Month')

plt.ylabel('Delay')

plt.show()



print ("The delays are at their highest between May and July, and in January and December too.")

print ("Thery're the least between August and September.")

print ("The high levels in January and December can be because of people travelling because of the winter break, and to be with their families for Christmas.")

print ("The low levels in AUgust and September can be accounted for by ")

#month against various delays
airline_delay_averages = delays[['AIRLINE', 'AIRLINE_DELAY', 

                                 'LATE_AIRCRAFT_DELAY', 'TOTAL_DELAY_AIRLINE']].groupby('AIRLINE').mean().reset_index()



plot_df = pd.melt(airline_delay_averages, id_vars = "AIRLINE", 

                  var_name = "DELAYS", value_name = "Average_Airline_Delays")

sb.factorplot(x = 'AIRLINE', y = 'Average_Airline_Delays',

              hue = 'DELAYS', data = plot_df, kind = 'bar')

plt.title('Airline Delays')

plt.xlabel('Airlines')

plt.ylabel('Average Airline Delays')

plt.show()



print (airline_dictionary['UA'] + " has the highest airline delay.")

print (airline_dictionary['F9'] + " has the highest Late Aircraft delay, resulting in this airlines having the highest total delay.")

print (airline_dictionary['HA'] + " and " + airline_dictionary['AS'] + " have achieved the lowest delays.")

#visualize delays vs airlines
#Calc speed, check for null values

print ("Airline speed can be calculated from the Distance between airports and the Departure and Arrival times.")

print ("The number of NAN entries is DISTANCE, DEPARTURE TIME and ARRIVAL TIME are:")

print (flights['DISTANCE'].isnull().sum())

print (flights['DEPARTURE_TIME'].isnull().sum())

print (flights['ARRIVAL_TIME'].isnull().sum())

#discard rows which have the latter two NULL

print ("Rows with Null values, and rows with Depature time equal to Arrival time are dropped.")

airline_speed = flights[['AIRLINE', 'DISTANCE', 

                         'DEPARTURE_TIME', 'ARRIVAL_TIME']]

airline_speed = airline_speed.dropna()

airline_speed = airline_speed.drop(airline_speed[airline_speed['ARRIVAL_TIME'] == airline_speed['DEPARTURE_TIME']].index)

airline_speed['FLIGHT_SPEED'] = airline_speed['DISTANCE'] / (airline_speed['ARRIVAL_TIME'] - airline_speed['DEPARTURE_TIME'])

airline_speed_average = airline_speed[['AIRLINE', 'FLIGHT_SPEED']].groupby('AIRLINE').mean().reset_index()

sb.barplot(x = "FLIGHT_SPEED", y = "AIRLINE", 

           data = airline_speed_average, palette = "Greens_d")

plt.title('Flight Speeds.')

plt.xlabel('Speed')

plt.ylabel('Airline')

plt.show()



print ("Clearly, " + airline_dictionary['HA'] + " is the slowest, while " + airline_dictionary['WN'] + " is the fastest.")

#visualize
frequent_day = flights[['DAY_OF_WEEK']].groupby('DAY_OF_WEEK').size().reset_index()

frequent_day.columns = ['DAY_OF_WEEK', 'COUNT']

sb.barplot(x = "DAY_OF_WEEK", y = "COUNT", 

           data = frequent_day, palette = "Blues_d")

plt.title('Frequently Flown - Day')

plt.xlabel('Day')

plt.ylabel('Frequency')

plt.show()

print ("The weekend has lesser traffic than the weekdays.")
frequent_month = flights[['MONTH']].groupby('MONTH').size().reset_index()

frequent_month.columns = ['MONTH', 'COUNT']

sb.barplot(x = "MONTH", y = "COUNT", 

           data = frequent_month, palette = "Blues_d")

plt.title('Frequently Flown - Month')

plt.xlabel('Month')

plt.ylabel('Frequency')

plt.show()

print ("Maximum traffic is present in July, and minimum in February.")

print ("The minimum in February could be because of the bad weather at that time of the year.")

print ("The trend of weather delay over the months, and its relation to the number of cancelled flights.")

weather_delays = flights[['MONTH', 'WEATHER_DELAY']].groupby('MONTH').mean().reset_index()

weather_delays['WEATHER_DELAY'] = weather_delays['WEATHER_DELAY'] * 1000

monthly_cancellation = flights[['MONTH', 'CANCELLED']][flights['CANCELLED'] == 1].groupby('MONTH').size().reset_index()

monthly_cancellation.columns = ['MONTH', 'CANCELLATIONS']

weather_cancellation = weather_delays.join(monthly_cancellation.set_index('MONTH'), on = 'MONTH')

p2 = plt.figure(figsize = (15, 10))

dim = p2.add_axes([0,0,1,1])

for column in weather_cancellation:

    if column is not 'MONTH':

        weather_cancellation[column].plot.line(x = 'MONTH', label = column)

dim.legend();

plt.title('Weather Delay - Cancellation')

plt.xlabel('Month')

plt.ylabel('Delay-Cancellation')

plt.show()

print ("As the weather delay increases, between December and February, so do the number of cancellations.")

print ("The weather delay decreases between June and August, and the number of cancellations also start dropping.")

print ("From this, it can be deduced that the cancellations that occured could have occured because of the bad weather that cause delays in the other flights.")

#plot month vs cancellation, relate to weather delays over months
origin_airport_weather = flights[['ORIGIN_AIRPORT', 'WEATHER_DELAY']]

origin_airport_weather.columns = ['IATA_CODE', 'WEATHER_DELAY']

dest_airport_weather = flights[['DESTINATION_AIRPORT', 'WEATHER_DELAY']]

dest_airport_weather.columns = ['IATA_CODE', 'WEATHER_DELAY']

airport_weather = pd.concat([origin_airport_weather, dest_airport_weather])

airport_weather = airport_weather.dropna()

airport_weather = airport_weather.join(airports[['IATA_CODE', 'STATE']].set_index('IATA_CODE'), on = 'IATA_CODE')

state_weather = airport_weather[['STATE', 'WEATHER_DELAY']].groupby('STATE').mean().reset_index()

state_weather = state_weather.sort_index(by = 'WEATHER_DELAY')

sb.barplot(x = "STATE", y = "WEATHER_DELAY", 

           data = state_weather[-15:], palette = "Reds_d")

plt.title('State - Weather Delay (Top Fifteen)')

plt.xlabel('State')

plt.ylabel('Weather Delay')

plt.show()

print ("The state with the most weather delays, implying the worst weather is " + state_weather.loc[state_weather['WEATHER_DELAY'].argmax()]['STATE'])

print ("And the state with the minimum is " + (state_weather.loc[state_weather['WEATHER_DELAY'].argmin()]['STATE']))

#location vs weather
# Airport Analysis and Ranking
incoming_traffic = flights[['ORIGIN_AIRPORT', 'DAY_OF_WEEK', 'MONTH']]

outgoing_traffic = flights[['DESTINATION_AIRPORT', 'DAY_OF_WEEK', 'MONTH']]

incoming_traffic.columns = ['AIRPORT', 'DAY_OF_WEEK', 'MONTH']

outgoing_traffic.columns = ['AIRPORT', 'DAY_OF_WEEK', 'MONTH']
airport_monthly_traffic = pd.concat([incoming_traffic[['AIRPORT', 'MONTH']], outgoing_traffic[['AIRPORT', 'MONTH']]])

airport_monthly_traffic = airport_monthly_traffic.fillna(0)

airport_monthly_traffic = airport_monthly_traffic.groupby(['AIRPORT', 'MONTH']).size().reset_index()

airport_monthly_traffic.columns = ['AIRPORT', 

                                   'MONTH', 'TRAFFIC']

airport_monthly_traffic = airport_monthly_traffic.sort_index(by = 'TRAFFIC')

airport_monthly_plot = airport_monthly_traffic[-100:].pivot_table(index = 'MONTH',

                                                           columns = 'AIRPORT', values = 'TRAFFIC')

airport_monthly_plot = airport_monthly_plot.fillna(0)

p4 = plt.figure(figsize = (15, 10))

dim = p4.add_axes([0,0,1,1])

for column in airport_monthly_plot:

    airport_monthly_plot[column].plot.line(x = 'MONTH', label = column)

dim.legend();  

plt.title('Trends in Airport Traffic over the months. (Top eleven)')

plt.xlabel('Month')

plt.ylabel('Traffic')

plt.show()



print ("Flights achieve maximum traffic between July and August.")

print ("Traffic for " + airport_monthly_traffic.loc[airport_monthly_traffic['TRAFFIC'].argmax()]['AIRPORT'] + " is the maximum.")

print ("It reaches the global maximum of " + str(airport_monthly_traffic['TRAFFIC'].max()) + " flights in September.")

print ("Traffic for " + airport_monthly_traffic.loc[airport_monthly_traffic['TRAFFIC'].argmin()]['AIRPORT'] + " is the minimum.")

print ("It reaches the global minimum of " + str(airport_monthly_traffic['TRAFFIC'].min()) + " flights in September.")

#Each airports distribution of monthly traffic
incoming_airport_traffic = incoming_traffic[['AIRPORT', 'DAY_OF_WEEK']].groupby('AIRPORT').size().reset_index()

incoming_airport_traffic.columns = ['AIRPORT', 'NO_OF_FLIGHTS']

incoming_airport_traffic = incoming_airport_traffic.sort_index(by = 'NO_OF_FLIGHTS')

p4 = plt.figure(figsize = (20, 10))

sb.barplot(x = "AIRPORT", y = "NO_OF_FLIGHTS",

           data = incoming_airport_traffic[-15:], palette = "Greens_d")

plt.title('No of incoming Flights. per Airport (Top Fifteen)')

plt.xlabel('Airport')

plt.ylabel('Count')

plt.show()

print ("Maximum incoming flights: ")

print (incoming_airport_traffic.loc[incoming_airport_traffic['NO_OF_FLIGHTS'].argmax()])

print ("Minimum incoming flights: " )

print (incoming_airport_traffic.loc[incoming_airport_traffic['NO_OF_FLIGHTS'].argmin()])

outgoing_airport_traffic = outgoing_traffic[['AIRPORT', 'DAY_OF_WEEK']].groupby('AIRPORT').size().reset_index()

outgoing_airport_traffic.columns = ['AIRPORT', 'NO_OF_FLIGHTS']

outgoing_airport_traffic = outgoing_airport_traffic.sort_index(by = 'NO_OF_FLIGHTS')

p4 = plt.figure(figsize = (20, 10))

sb.barplot(x = "AIRPORT", y = "NO_OF_FLIGHTS", 

           data = outgoing_airport_traffic[-15:], palette = "Greens_d")

plt.title('No of outgoing Flights. per Airport (Top Fifteen)')

plt.xlabel('Airport')

plt.ylabel('Count')

plt.show()

print ("Maximum outgoing flights: ")

print (outgoing_airport_traffic.loc[outgoing_airport_traffic['NO_OF_FLIGHTS'].argmax()])

print ("Minimum outgoing flights: ")

print (outgoing_airport_traffic.loc[outgoing_airport_traffic['NO_OF_FLIGHTS'].argmin()])





print ("Therefore, " + outgoing_airport_traffic.loc[outgoing_airport_traffic['NO_OF_FLIGHTS'].argmax()]['AIRPORT'] + " has the highest traffic.")
source_airport_delay_averages = delays[['ORIGIN_AIRPORT', 'TOTAL_DELAY_SOURCE_AIRPORT']].groupby('ORIGIN_AIRPORT').mean().reset_index()

dest_airport_delay_averages = delays[['DESTINATION_AIRPORT', 'TOTAL_DELAY_DEST_AIRPORT']].groupby('DESTINATION_AIRPORT').mean().reset_index()



in_traffic_delay = incoming_airport_traffic

in_traffic_delay.columns = ['ORIGIN_AIRPORT', 'NO_OF_FLIGHTS']

in_traffic_delay = in_traffic_delay.join(source_airport_delay_averages.set_index('ORIGIN_AIRPORT'),

                                         on = 'ORIGIN_AIRPORT' )

fig = sb.lmplot('NO_OF_FLIGHTS', 'TOTAL_DELAY_SOURCE_AIRPORT', 

          data = in_traffic_delay, fit_reg = False, hue = 'ORIGIN_AIRPORT', legend = False)

plt.title('Incoming Traffic vs Delay Incurred')

plt.xlabel('No of Flights')

plt.ylabel('Source Airport Delay')

plt.show()

print ("It can be observed that " + in_traffic_delay.loc[in_traffic_delay['NO_OF_FLIGHTS'].argmax()]['ORIGIN_AIRPORT'] + " has the maximum number of incoming flights, and average amount of delay. It is relatively capable of handling the traffic.")
out_traffic_delay = outgoing_airport_traffic

out_traffic_delay.columns = ['DESTINATION_AIRPORT', 'NO_OF_FLIGHTS']

out_traffic_delay = out_traffic_delay.join(dest_airport_delay_averages.set_index('DESTINATION_AIRPORT'),

                                           on = 'DESTINATION_AIRPORT' )

sb.lmplot('NO_OF_FLIGHTS', 'TOTAL_DELAY_DEST_AIRPORT',

          data = out_traffic_delay, fit_reg = False, hue = 'DESTINATION_AIRPORT', legend = False)

plt.title('Outgoing Traffic vs Delay Incurred')

plt.xlabel('No of Flights')

plt.ylabel('Destination Airport Delay')

plt.show()

print (out_traffic_delay.loc[out_traffic_delay['NO_OF_FLIGHTS'].argmax()]['DESTINATION_AIRPORT'] + " also has the maximum number of outgoing flights, and average amount of delay. It is capable of handling the outgoing traffic too.")

print ("On the other hand, " + in_traffic_delay.loc[in_traffic_delay['TOTAL_DELAY_SOURCE_AIRPORT'].argmax()]['ORIGIN_AIRPORT'] + " has a low Number of (incoming and outgoing) Flights, but maximum delay. Therefore, it isn't as efficient.")



#Map traffic against delays to see how well handled.
airport_weather_averages = airport_weather[['STATE', 'IATA_CODE', 'WEATHER_DELAY']].groupby(['STATE', 'IATA_CODE']).mean().reset_index()

airport_weather_averages = airport_weather_averages.loc[:100]

airport_weather_averages.columns = ['STATE', 'AIRPORT', 'WEATHER_DELAY']

airport_weather_averages = airport_weather_averages.sort_index(by = 'WEATHER_DELAY')

sb.factorplot(x = 'STATE', y = 'WEATHER_DELAY',

              hue = 'AIRPORT', data = airport_weather_averages[-15:], kind = 'bar')

plt.title('State-wise weather delays incurred by airports')

plt.xlabel('State')

plt.ylabel('Delay')

plt.show()

print ("In State AK, from the airports JNU, OTZ, SCC, it can be seen that JNU has the least weather delays. So, in AK, JNU is the most efficient airport at handling bad weather conditions.")

print ("Similarly, in IA, CID is much better than DBQ at handling bad weather conditions.")

#map averages state wise to show which handles the worst weather the best
airport_cancellation = flights[['ORIGIN_AIRPORT', 'CANCELLED']][flights['CANCELLED'] == 1].groupby('ORIGIN_AIRPORT').size().reset_index()

airport_cancellation.columns = ['ORIGIN_AIRPORT', 'CANCELLED']

airport_cancellation = airport_cancellation.sort_index(by = 'CANCELLED')

sb.barplot(x = "ORIGIN_AIRPORT", y = "CANCELLED", 

           data = airport_cancellation[-15:], palette = "Blues_d")

plt.title('Airport Cancellations (Top Fifteen)')

plt.xlabel('Airport')

plt.ylabel('Cancellations')

plt.show()

print ("Maximum number of cancellations:")

print ("By " + airport_cancellation.loc[airport_cancellation['CANCELLED'].argmax()]['ORIGIN_AIRPORT'])

print ("Cancellations: " + str(airport_cancellation.loc[airport_cancellation['CANCELLED'].argmax()]['CANCELLED']))

print ("Minimum number of cancellations:")

print ("By " + airport_cancellation.loc[airport_cancellation['CANCELLED'].argmin()]['ORIGIN_AIRPORT'])

print ("Cancellations: " + str(airport_cancellation.loc[airport_cancellation['CANCELLED'].argmin()]['CANCELLED']))



#airport vs cancelllation