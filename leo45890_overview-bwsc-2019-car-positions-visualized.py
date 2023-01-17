import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # plotting

import plotly.express as px # fancy plotting



# import data

positions = pd.read_csv("../input/bwsc-2019-car-positions/bwsc2019-car-positions.csv")
positions.gps_when = pd.to_datetime(positions.gps_when)

positions.drop_duplicates(subset=["gps_when", "car_name"], inplace=True)
positions.head()
positions.shape
positions.describe()
print("Participating countries: {}".format(positions.country.unique()))



print("Number of participating teams: {}".format(len(positions.name.unique())))



print("Participating cars: {}".format(positions.car_name.unique()))

positions.groupby("class_id").car_name.unique()
reduced_data = positions.sample(n=2000)



fig = px.scatter_geo(reduced_data, lat="lat", lon="lng")

fig.show()
# filter class_id (5 = challenger, 6 = cruiser, 7 = adventure)

positions_filtered = positions[positions.class_id == 5]



fig, ax = plt.subplots(figsize=(18,10))

for label, df in positions_filtered.groupby("car_name"):

    df.plot(x="gps_when", y="dist_adelaide", ax=ax, label=label)
def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    meters = 6367.0e3 * c

    return meters

car = "BluePoint"



# clean DataFrame of positions for the selected car

car_position = positions[positions.car_name == car].sort_values(by="gps_when").reset_index(drop=True)



# calculate distance between GPS points and velocity

car_position["gps_dist"] = haversine_np(car_position.lat.shift(), car_position.lng.shift(), car_position.loc[1:, 'lat'], car_position.loc[1:, 'lng']).fillna(0)

dt = car_position.gps_when.diff().fillna(pd.Timedelta(seconds=1000)).map(pd.Timedelta.total_seconds)

car_position["velocity"] = car_position.gps_dist / dt



fig, ax = plt.subplots(figsize=(18,10))

car_position.plot(x="gps_when", y="velocity", ax=ax)



total_distance = car_position.gps_dist.sum()



print("Distance driven: {}km".format(total_distance/1.0e3))