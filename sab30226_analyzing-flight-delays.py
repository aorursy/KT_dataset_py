import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
ROWS_TO_READ = 10000

#ap = pd.read_csv('airports.csv')

#al = pd.read_csv('airlines.csv')

df = pd.read_csv('../input/flight-delays/flights.csv', nrows=ROWS_TO_READ)
df.head()
df['DELAYED'] = df['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 0 else 0)

reasons = df[['DIVERTED', 'CANCELLED', 'DELAYED']]

reasons.head()
reasons['ON_TIME'] = (reasons['DIVERTED'] + reasons['CANCELLED'] + reasons['DELAYED']).apply(lambda x: 1 if x > 0 else 0)

reasons.head()
display(sns.barplot(reasons.sum(), reasons.columns))

precentage_of_delayed_flights = round((reasons['DELAYED'].sum() / len(reasons)) * 100, 2)

print(f"Total delayed flight: {precentage_of_delayed_flights}%")

display(reasons.sum())
delays = df[df['DELAYED'] == True]

delays = delays[['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]

delays.fillna(0, inplace=True)

delays.astype(float)

delays.head()
display(delays.sum())

display(sns.barplot(delays.sum(), delays.columns))

#precentage_of_delayed_flights = round((reasons['DELAYED'].sum() / len(reasons)) * 100, 2)

#print(f"Total delayed flight: {precentage_of_delayed_flights}%")

print("Most of the delays are caused by the airlines")
display(df[['FLIGHT_NUMBER', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE', 'ELAPSED_TIME', 'ARRIVAL_DELAY']].head())
df.describe()
plt.hist(df['ARRIVAL_DELAY'])

plt.hist(df['DEPARTURE_DELAY'])

plt.show()
numerical = df.select_dtypes(include=['float64'])

numerical = numerical[['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'WHEELS_OFF',

       'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',

       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]

sns.heatmap(numerical.corr())
display(sns.scatterplot(df['ARRIVAL_DELAY'], df['AIR_SYSTEM_DELAY'], alpha=0.2))
display(sns.scatterplot(df['ARRIVAL_DELAY'], df['AIRLINE_DELAY'], alpha=0.2))