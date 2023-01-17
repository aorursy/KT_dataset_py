%matplotlib inline
import sys

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import geopy.distance



from nose.tools import *

from mpl_toolkits.basemap import Basemap
weather_station_locations = pd.read_csv("../input/weatherww2/Weather Station Locations.csv", index_col = 0, header=0,

    names=["wban", "name", "state_country_id", "str_latitude", "str_longitude", "elev", "latitude", "longitude"])

weather_station_locations.shape
weather_station_locations.sample(5)
summary_of_weather = pd.read_csv("../input/weatherww2/Summary of Weather.csv", index_col = 0, header=0,

    names=["sta", "date", "precip", "wind_gust_spd", "max_temp", "min_temp", "mean_temp", "snowfall",

           "poor_weather", "yr", "mo", "da", "prcp", "dr", "spd", "max", "min", "mea", "snf", "snd",

           "ft", "fb", "fti", "ith", "pgt", "tshdsbrsgf", "sd3", "rhx", "rhn", "rvg", "wte"], low_memory=False)

summary_of_weather.shape
summary_of_weather.sample(5)
aerial_bombing_operations = pd.read_csv("../input/world-war-ii/operations.csv", index_col = 0, header=0,

    names=["mission_id", "mission_date", "theater_of_operations", "country", "air_force", "unit_id",

           "aircraft_series", "callsign", "mission_type", "takeoff_base", "takeoff_location", "takeoff_latitude",

           "takeoff_longitude", "target_id", "target_country", "target_city", "target_type", "target_industry",

           "target_priority", "target_latitude", "target_longitude", "altitude_hundreds_of_feet", "airborne_aircraft",

           "attacking_aircraft", "bombing_aircraft", "aircraft_returned", "aircraft_failed", "aircraft_damaged",

           "aircraft_lost", "high_explosives", "high_explosives_type", "high_explosives_weight_pounds",

           "high_explosives_weight_tons", "incendiary_devices", "incendiary_devices_type",

           "incendiary_devices_weight_pounds", "incendiary_devices_weight_tons", "fragmentation_devices",

           "fragmentation_devices_type", "fragmentation_devices_weight_pounds", "fragmentation_devices_weight_tons",

           "total_weight_pounds", "total_weight_tons", "time_over_target", "bomb_damage_assessment", "source_id"], low_memory=False)

aerial_bombing_operations.shape
aerial_bombing_operations.sample(5)
weather_station_locations.info()
weather_station_locations = weather_station_locations[["latitude", "longitude"]]



assert weather_station_locations.shape == (161, 2)
weather_station_locations = weather_station_locations[((weather_station_locations.latitude <= 90) &

                                (weather_station_locations.latitude >= -90)) &

                                ((weather_station_locations.longitude <= 180) &

                                (weather_station_locations.longitude >= -180))]



assert weather_station_locations.shape == (161, 2)
summary_of_weather.info()
summary_of_weather = summary_of_weather[["date", "precip", "snowfall", "poor_weather"]]

summary_of_weather.info()
summary_of_weather.date =  pd.to_datetime(summary_of_weather.date, format = "%Y-%m-%d")

summary_of_weather.info()
summary_of_weather.precip[(pd.to_numeric(summary_of_weather.precip, errors='coerce').notnull()) == False].unique()
summary_of_weather.loc[(summary_of_weather.precip == "T"), "precip"] = 0.00

summary_of_weather.precip = summary_of_weather.precip.astype(float)

summary_of_weather.info()
summary_of_weather.snowfall.unique()
summary_of_weather.loc[(summary_of_weather.snowfall == "#VALUE!"), "snowfall"] = 0.00

summary_of_weather.snowfall = summary_of_weather.snowfall.fillna(0.00)

summary_of_weather.snowfall = summary_of_weather.snowfall.astype(float)

summary_of_weather.info()
summary_of_weather.poor_weather.unique()
summary_of_weather.groupby("poor_weather")["poor_weather"].count()
summary_of_weather.poor_weather = summary_of_weather.poor_weather.apply(lambda x: str(x).count("1"))

summary_of_weather.poor_weather = summary_of_weather.poor_weather.astype("int64")

summary_of_weather.poor_weather.unique()
summary_of_weather.precip[summary_of_weather.precip > 0].describe(percentiles = [.5, .7, .9, .98]),\

summary_of_weather.snowfall[summary_of_weather.snowfall > 0].describe(percentiles = [.5, .7, .9, .98])
summary_of_weather.groupby("poor_weather")["poor_weather"].count()
mask = ((summary_of_weather.precip > 4) & (summary_of_weather.precip <= 9) & (summary_of_weather.poor_weather < 1))

summary_of_weather.loc[mask, "poor_weather"] = 1

mask = ((summary_of_weather.precip > 9) & (summary_of_weather.precip <= 27) & (summary_of_weather.poor_weather < 2))

summary_of_weather.loc[mask, "poor_weather"] = 2

mask = ((summary_of_weather.precip > 27) & (summary_of_weather.precip <= 63) & (summary_of_weather.poor_weather < 3))

summary_of_weather.loc[mask, "poor_weather"] = 3

mask = ((summary_of_weather.precip > 63) & (summary_of_weather.poor_weather < 4))

summary_of_weather.loc[mask, "poor_weather"] = 4
summary_of_weather.groupby("poor_weather")["poor_weather"].count()
mask = ((summary_of_weather.snowfall > 8) & (summary_of_weather.snowfall <= 14) & (summary_of_weather.poor_weather < 1))

summary_of_weather.loc[mask, "poor_weather"] = 1

mask = ((summary_of_weather.snowfall > 14) & (summary_of_weather.snowfall <= 30) & (summary_of_weather.poor_weather < 2))

summary_of_weather.loc[mask, "poor_weather"] = 2

mask = ((summary_of_weather.snowfall > 30) & (summary_of_weather.snowfall <= 61) & (summary_of_weather.poor_weather < 3))

summary_of_weather.loc[mask, "poor_weather"] = 3

mask = ((summary_of_weather.snowfall > 61) & (summary_of_weather.poor_weather < 4))

summary_of_weather.loc[mask, "poor_weather"] = 4
summary_of_weather.groupby("poor_weather")["poor_weather"].count()
aerial_bombing_operations.info()
aerial_bombing_operations = aerial_bombing_operations[["mission_date", "target_latitude",

    "target_longitude", "aircraft_returned", "aircraft_failed", "aircraft_damaged", "aircraft_lost"]]

aerial_bombing_operations.shape
aerial_bombing_operations = aerial_bombing_operations.dropna(subset=["target_latitude", "target_longitude"])

aerial_bombing_operations.info()
aerial_bombing_operations = aerial_bombing_operations[((aerial_bombing_operations.target_latitude <= 90) &\

                                   (aerial_bombing_operations.target_latitude >= -90)) & \

                                         ((aerial_bombing_operations.target_longitude <= 180) &\

                                   (aerial_bombing_operations.target_longitude >= -180))]

aerial_bombing_operations.shape
aerial_bombing_operations.mission_date =  pd.to_datetime(aerial_bombing_operations.mission_date, format = "%m/%d/%Y")

aerial_bombing_operations.info()
plt.hist(summary_of_weather.poor_weather, bins = 5)

plt.title("Weather conditions distribution")

plt.xlabel("Poor weather value")

plt.ylabel("Count")

plt.show()
plt.hist(aerial_bombing_operations.mission_date.dt.year, bins = 7)

plt.title("Operations by year distribution")

plt.xlabel("Year")

plt.ylabel("Count")

plt.show()
def drawgeolocations(longitude, latitude, loc_size, loc_color, label):

    figure = plt.figure(figsize = (15, 12))

    current_axis = figure.add_subplot(111)

    m = Basemap(projection = "merc", llcrnrlat = -80, urcrnrlat = 80,

    llcrnrlon = -180, urcrnrlon = 180)

    

    locations_lon, locations_lat = m(longitude.tolist(), latitude.tolist())

    

    m.drawcoastlines(ax = current_axis)

    m.fillcontinents(color = "coral", lake_color = "aqua")

    m.drawparallels(np.arange(-90, 91, 30))

    m.drawmeridians(np.arange(-180, 181, 60))

    m.drawmapboundary(fill_color = "aqua")

    m.scatter(locations_lon, locations_lat, color = loc_color, s = loc_size, zorder = 2)

    plt.xlabel(label)

    plt.show()
drawgeolocations(weather_station_locations.longitude, weather_station_locations.latitude, 

                 30, "green", "Location of weather stations")
drawgeolocations(aerial_bombing_operations.target_longitude, aerial_bombing_operations.target_latitude, 

                 30, "red", "Locations of air attacks")
summary_of_weather_with_loc = summary_of_weather.join(weather_station_locations)

summary_of_weather_with_loc.info()
def get_closest_weatherstation(lat, lot, weather_stations):

    min_distance = sys.maxsize

    min_station_index = -1

    for index, station in weather_station_locations.iterrows():

        distance = geopy.distance.distance((lat, lot), (station["latitude"], station["longitude"]))

        if min_distance > distance:

            min_distance = distance

            min_station_index = index

            

    return min_station_index, min_distance

    

assert get_closest_weatherstation(40.23333333, 18.08333333, weather_station_locations) == (34111, 0)
# Very slow operation - over 2 hours. Result is saved to ...data/operation_to_sta_mapping.csv



#lambdafunc = lambda x: get_closest_weatherstation(x["target_latitude"], x["target_longitude"], weather_station_locations)

#operation_to_sta_mapping = aerial_bombing_operations.apply(lambdafunc, axis=1)

#csv_operation_to_sta_mapping_df = pd.DataFrame(operation_to_sta_mapping, columns = ["tuple"])["tuple"].apply(pd.Series)

#csv_operation_to_sta_mapping_df.columns = ["sta", "sta_distance"]

#csv_operation_to_sta_mapping_df.sta_distance = csv_operation_to_sta_mapping_df.sta_distance.astype(str).str[:-3].astype(float)

#csv_operation_to_sta_mapping_df.to_csv("data/operation_to_sta_mapping.csv")



operation_to_sta_mapping = pd.read_csv("../input/ww2-aerial-operation-to-closest-ww2-weather-statio/operation_to_sta_mapping.csv", index_col = 0, header=0)

operation_to_sta_mapping.head(5)
operations_with_weather_sta = aerial_bombing_operations.join(operation_to_sta_mapping)

operations_with_weather_sta.head(5)
operations_with_weather_sta = operations_with_weather_sta.reset_index().set_index(["sta", "mission_date"])

operations_with_weather_sta.index.names = ["sta", "date"]

operations_with_weather_sta.head(5)
summary_of_weather = summary_of_weather.reset_index().set_index(["sta", "date"])

summary_of_weather.head(5)
operations_with_weather = operations_with_weather_sta.merge(summary_of_weather, left_index=True, right_on=["sta", "date"])

operations_with_weather = operations_with_weather.reset_index().set_index("mission_id")

operations_with_weather.head(5)
def plot_operations_weather_distr_by_sta_distance(operations, distance):

    operations_by_sta_distance = operations[operations.sta_distance < distance]

    operations_by_weather = operations_by_sta_distance.groupby("poor_weather")["poor_weather"].count()

    operations_by_weather.plot.bar()

    plt.title("Operations by weather condition with distance \n to closest weather station < " + str(distance))

    plt.xlabel("Poor weather value")

    plt.ylabel("Operations count")

    plt.show()
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 2)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 10)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 50)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 100)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 500)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 1000)
plot_operations_weather_distr_by_sta_distance(operations_with_weather, 10000)
worst_weather_operations = operations_with_weather[operations_with_weather.poor_weather == 4]

drawgeolocations(worst_weather_operations.target_longitude, worst_weather_operations.target_latitude, 30, "blue", 

                 "Locations of air attacks poor_weather = 4")
worst_weather_operations = operations_with_weather[operations_with_weather.poor_weather == 3]

drawgeolocations(worst_weather_operations.target_longitude, worst_weather_operations.target_latitude, 30, "blue",

                 "Locations of air attacks poor_weather = 3")
worst_weather_operations = operations_with_weather[operations_with_weather.poor_weather == 2]

drawgeolocations(worst_weather_operations.target_longitude, worst_weather_operations.target_latitude, 30, "blue",

                 "Locations of air attacks poor_weather = 2")
worst_weather_operations = operations_with_weather[operations_with_weather.poor_weather == 1]

drawgeolocations(worst_weather_operations.target_longitude, worst_weather_operations.target_latitude, 30, "blue",

                 "Locations of air attacks poor_weather = 1")
worst_weather_operations = operations_with_weather[operations_with_weather.poor_weather == 0]

drawgeolocations(worst_weather_operations.target_longitude, worst_weather_operations.target_latitude, 30, "blue",

                 "Locations of air attacks poor_weather = 0")