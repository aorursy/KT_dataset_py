import numpy as np

import pandas as pd

import glob

import seaborn as sns

import networkx as nx

import tensorflow as tf

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/historical-flight-and-weather-data/*.csv") ])
df.head()
df.hist(figsize=(20,20)); # Tip: put a semicolon at the end of the line to avoid printing a bunch of text output.
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
num_flights_spatial = num_flights.reset_index().merge(airports[["IATA", "Latitude", "Longitude"]], how="inner", left_on="origin_airport", right_on="IATA")

num_flights_spatial["lat_origin"] = num_flights_spatial["Latitude"]

num_flights_spatial["lon_origin"] = num_flights_spatial["Longitude"]

num_flights_spatial.drop(['IATA', 'Latitude', "Longitude"], inplace=True, axis=1)



num_flights_spatial = num_flights_spatial.merge(airports[["IATA", "Latitude", "Longitude"]], how="inner", left_on="destination_airport", right_on="IATA")

num_flights_spatial["lat_destination"] = num_flights_spatial["Latitude"]

num_flights_spatial["lon_destination"] = num_flights_spatial["Longitude"]

num_flights_spatial.drop(['IATA', 'Latitude', "Longitude"], inplace=True, axis=1)





num_flights_spatial.head()
num_flights_spatial.describe()
from sklearn.metrics.pairwise import haversine_distances

from math import radians





def great_circle(row):

    d = haversine_distances([[radians(row.lat_origin), radians(row.lon_origin)], [radians(row.lat_destination), radians(row.lon_destination)]])

    d = d * 6371000/1000

    return d[0][1] # The haversine distances function returns a 2-d array for some reason.



num_flights_spatial["great_circle_km"] = num_flights_spatial.apply(great_circle, axis=1)



num_flights_spatial["great_circle_km"].describe()
df_spatial = df_net_stats.merge(num_flights_spatial, how="inner", on=["origin_airport","destination_airport"])



df_spatial.head()
df_spatial[["departure_delay","arrival_delay", "great_circle_km", "flight_number_y"]].corr()
df_spatial.columns
feature_columns = []

feature_data = {}



def add_categorical_column(df, key):

    feat_col = tf.feature_column.categorical_column_with_vocabulary_list(

        key=key, vocabulary_list=df[key].unique())

    feat_col = tf.feature_column.indicator_column(feat_col)

    feature_columns.append(feat_col)

    feature_data[key] = np.array(df[key])

    

add_categorical_column(df_spatial, "carrier_code")



# add more here...





feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
label_data = np.array(df_spatial["arrival_delay"])
# These functions are adapted from the machine learning crash course.



def create_model(learning_rate, feature_layer):

  # Most simple tf.keras models are sequential.

  model = tf.keras.models.Sequential()



  # Add the layer containing the feature columns to the model.

  model.add(feature_layer)

    

  # Add one linear layer to the model to yield a simple linear regressor.

  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))    

    

  # Construct the layers into a model that TensorFlow can execute.

  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),

                loss="mean_squared_error",

                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    

  return model



def plot_the_loss_curve(epochs, rmse):

  """Plot a curve of loss vs. epoch."""



  plt.figure()

  plt.xlabel("Epoch")

  plt.ylabel("Root Mean Squared Error")



  plt.plot(epochs, rmse, label="Loss")

  plt.legend()

  plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])

  plt.show()  
# The following variables are the hyperparameters.

learning_rate = 0.05

epochs = 10

batch_size = 1000



# Create and compile the model's topography.

model = create_model(learning_rate, feature_layer)
# Train the model on the training set.

history = model.fit(x=feature_data, y=label_data, batch_size=batch_size,

                  epochs=epochs, shuffle=True, steps_per_epoch=100)

# We wouldn't normally use the 'steps_per_epoch' argument, but we're using it

# here so the training goes faster. (Basically we're only training on part of the data).



# The list of epochs is stored separately from the rest of history.

epochs = history.epoch



# Isolate the mean absolute error for each epoch.

hist = pd.DataFrame(history.history)

rmse = hist["root_mean_squared_error"]
model.summary()
plot_the_loss_curve(epochs, rmse)
plot_data = {key : feature_data[key][0:1000] for key in feature_data}

y_prediction = model.predict(plot_data)
plt.scatter(label_data[0:1000], y_prediction);