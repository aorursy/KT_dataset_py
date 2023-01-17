import numpy as np
import pandas as pd
import glob
import seaborn as sns
import networkx as nx

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/historical-flight-and-weather-data/*.csv") ])
df.head()
df.dtypes
df.hist(figsize=(20,20)); # Tip: put a semicolon at the end of the line to avoid printing a bunch of text output.
df.shape
df.dtypes
(df.arrival_delay > 0).sum() / df.shape[0]
(df.arrival_delay > 30).sum() / df.shape[0]
(df.arrival_delay > 60).sum() / df.shape[0]
(df.departure_delay > 0).sum() / df.shape[0]
((df.arrival_delay > 0) & (df.departure_delay > 0)).sum() / df.shape[0]
df.cancelled_code.value_counts()
(df.cancelled_code != "N").sum() / df.shape[0]
df_cancel = df[df.cancelled_code != "N"]
df_cancel.hist(figsize=(20,20)); 
df_nocancel = df[df.cancelled_code == "N"]
df_nocancel.hist(figsize=(20,20)); 
print(df_cancel.HourlyWindSpeed_x.mean(), df_cancel.HourlyWindSpeed_x.median(), df_cancel.HourlyWindSpeed_x.max())
print(df_nocancel.HourlyWindSpeed_x.mean(), df_nocancel.HourlyWindSpeed_x.median(), df_nocancel.HourlyWindSpeed_x.max())
num_flights = df.groupby(by=["origin_airport", "destination_airport"]).count()['flight_number']

num_flights.head()
num_flights.reset_index().head()
g = nx.DiGraph()

for _, edge in num_flights.reset_index().iterrows():
    g.add_edge(edge['origin_airport'], edge['destination_airport'], weight=edge['flight_number'])
nx.draw(g)
deg_cen = nx.degree_centrality(g)

airport, dc = [], []
for k in deg_cen:
    airport.append(k)
    dc.append(deg_cen[k])

data = {"airport": airport, "deg_cen": dc}
    
df_deg_cen = pd.DataFrame(data)
df_deg_cen.set_index("airport", inplace=True)

df_deg_cen.head()
bet_cen = nx.betweenness_centrality(g, weight="weight")

airport, bc = [], []
for k in bet_cen:
    airport.append(k)
    bc.append(bet_cen[k])

data = {"airport": airport, "bet_cen": bc}
    
df_bet_cen = pd.DataFrame(data)
df_bet_cen.set_index("airport", inplace=True)

df_bet_cen.head()
net_stats = df_deg_cen
net_stats["bet_cen"] = df_bet_cen.bet_cen
net_stats.reset_index(inplace=True)

net_stats.head()
df_net_stats = df.merge(net_stats, left_on="origin_airport", right_on="airport")

df_net_stats["origin_bet_cen"] = df_net_stats["bet_cen"]
df_net_stats["origin_deg_cen"] = df_net_stats["deg_cen"]
df_net_stats.drop(["airport", "deg_cen", "bet_cen"], inplace=True, axis=1)

df_net_stats.head()
df_net_stats = df_net_stats.merge(net_stats, left_on="destination_airport", right_on="airport")

df_net_stats["destination_bet_cen"] = df_net_stats["bet_cen"]
df_net_stats["destination_deg_cen"] = df_net_stats["deg_cen"]
df_net_stats.drop(["airport", "deg_cen", "bet_cen"], inplace=True, axis=1)

df_net_stats.head()
df_net_stats[["arrival_delay", "destination_bet_cen","destination_deg_cen", "origin_bet_cen","origin_deg_cen"]].corr()
! wget https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
cols = ['Airport ID', #Unique OpenFlights identifier for this airport.
'Name', # Name of airport. May or may not contain the City name.
'City', # Main city served by airport. May be spelled differently from Name.
'Country', # Country or territory where airport is located. See Countries to cross-reference to ISO 3166-1 codes.
'IATA', # 3-letter IATA code. Null if not assigned/unknown.
'ICAO', # 4-letter ICAO code. Null if not assigned.
'Latitude', # Decimal degrees, usually to six significant digits. Negative is South, positive is North.
'Longitude', # Decimal degrees, usually to six significant digits. Negative is West, positive is East.
'Altitude', # In feet.
'Timezone', # Hours offset from UTC. Fractional hours are expressed as decimals, eg. India is 5.5.
'DST', # Daylight savings time. One of E (Europe), A (US/Canada), S (South America), O (Australia), Z (New Zealand), N (None) or U (Unknown). See also: Help: Time
'Tz', # database time zone	Timezone in "tz" (Olson) format, eg. "America/Los_Angeles".
'Type', # Type of the airport. Value "airport" for air terminals, "station" for train stations, "port" for ferry terminals and "unknown" if not known. In airports.csv, only type=airport is included.
'Source', # Source of this data. "OurAirports" for data sourced from OurAirports, "Legacy" for old data not matched to OurAirports (mostly DAFIF), "User" for unverified user contributions. In airports.csv, only source=OurAirports is included.
]

airports = pd.read_csv("airports.dat", names=cols)

airports.head()

num_flights_spatial.describe()






