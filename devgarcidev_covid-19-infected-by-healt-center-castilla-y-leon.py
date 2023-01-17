import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
file = "/kaggle/input/infected-by-health-center-11052020/tasa-enfermos-acumulados-por-areas-de-salud.csv"

data = pd.read_csv(file, encoding = "ISO-8859-1", delimiter=";")

data.head()

## data.describe()

data.columns
import math

import pandas as pd

import geopandas as gpd

from learntools.geospatial.tools import geocode





import folium 

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

from learntools.core import binder

binder.bind(globals())



## Function for show map

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')



# Create a base map

map_marker = folium.Map(location=[42.36,-5.34], zoom_start=8)



## Split data position in two columns, It is necessary to obtain the coordinates.

data['Latitud'], data['Longitud'] = data['POSICION'].str.split(',').str



for idx, row in data[data['PROVINCIA']=="Leon"].iterrows():

    Marker([row['Latitud'], row['Longitud']]).add_to(map_marker)



# Show the map

embed_map(map_marker, 'q_2.html')

### Create the map with folium.plugins.MarkerCluster()

map_MarkerCluster = folium.Map(location=[42.36,-5.34], tiles='cartodbpositron', zoom_start=8)



# Add points to the map, only for Leon province

mc = MarkerCluster()

for idx, row in data[data['PROVINCIA']=="Leon"].iterrows():

    mc.add_child(Marker([row['Latitud'], row['Longitud']]))

        

map_MarkerCluster.add_child(mc)



# Display the map

map_MarkerCluster
# Create a base map with folium.plugins.HeatMap().

map_HeatMap = folium.Map(location=[42.36,-5.34], tiles='cartodbpositron', zoom_start=7)



# Add a heatmap to the base map

HeatMap(data=data[['Latitud', 'Longitud']], radius=10).add_to(map_HeatMap)



# Display the map

map_HeatMap
sum_max_by_center = data.groupby(['CENTRO']).TOTALENFERMEDAD.agg([max,sum])



print("Sum and max infected by healthy center: ")

print (sum_max_by_center)

print("-----------------------------")



print("Sum and max infected from C.S. TROBAJO DEL CAMINO healthy center: ")

print(sum_max_by_center.loc['C.S. TROBAJO DEL CAMINO'])

print("-----------------------------")



print("Sum infected by healthy center: ")

print(data.groupby(['CENTRO']).apply(lambda df: df.TOTALENFERMEDAD.sum()))

print("-----------------------------")

from sklearn.tree import DecisionTreeRegressor



## First case

data_bycenterTrobajo = data.loc[data.CENTRO == 'C.S. TROBAJO DEL CAMINO']

y = data_bycenterTrobajo.TOTALENFERMEDAD

features = ['PCR_REALIZADOS','PCR_POSITIVOS','PCR_POSITIVOS_7DIAS','Latitud', 'Longitud']

X = data_bycenterTrobajo[features]



## fill data Nan

X = X.fillna(X.mean())



## Create a model DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)



print("Making predictions for the following 5: ")

print(X.head())

print("The predictions are")

print(model.predict(X.head()))





## Second case

file = "/kaggle/input/infected-by-health-center-11052020/datavalidationformodel.csv"

data_validation = pd.read_csv(file, encoding = "ISO-8859-1", delimiter=";")

data_validation['Latitud'], data_validation['Longitud'] = data_validation['POSICION'].str.split(',').str

X2 = data_validation[features]

data_validation.head()

model.predict(X2)



## Mirar la MAE

from sklearn.metrics import mean_absolute_error



predicted_infected = model.predict(X2)



data_bycenterTrobajo2 = data_validation.loc[data_validation.CENTRO == 'C.S. TROBAJO DEL CAMINO']

y2 = data_bycenterTrobajo2.TOTALENFERMEDAD



mean_absolute_error(y2, predicted_infected) ## 10.857142857142858 There is an error 10. (TOTALENFERMEDAD)

## It means that on average in each prediction there is a variation of 10 in the number of infected.





## MAE 

from sklearn.metrics import mean_absolute_error



predicted_infected = model.predict(X)

mean_absolute_error(y, predicted_infected)



## Split model

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)



# Define model

split_model = DecisionTreeRegressor()

# Fit model

split_model.fit(train_X, train_y)



# get predicted infected on validation data

val_predictions = split_model.predict(val_X)## val_X has 21 registers

print(mean_absolute_error(val_y, val_predictions))





# Use model split_model with data X2 and y2 for we see the difference the MAE

val_predictions = split_model.predict(X2)## val_X has 21 registers

print(mean_absolute_error(y2, val_predictions))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

infected_predictions = forest_model.predict(val_X)

print(mean_absolute_error(val_y, infected_predictions))