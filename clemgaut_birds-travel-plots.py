import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import datetime
df = pd.read_csv(r"../input/migration_original.csv")



# We add new columns for year and month that we'll use in next plots

time_format = "%Y-%m-%d %H:%M:%S.%f"

df["month"] = df["timestamp"].apply(lambda x : datetime.datetime.strptime(x, time_format).strftime("%B"))

df["year"] = df["timestamp"].apply(lambda x : datetime.datetime.strptime(x, time_format).year)



# We select the birds with more than 500 data points to have interesting travels

interesting_birds = df.groupby(["tag-local-identifier"])["tag-local-identifier"].filter(lambda x: len(x) > 500).unique()
# We select a bird from the interesting ones, any should do

bird_df = df.loc[df['tag-local-identifier'] == interesting_birds[-2]]



# We create the map with an appropriate bounding box

map = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=68,llcrnrlon=-10,urcrnrlon=65,lat_ts=20,resolution='i')

map.drawcoastlines()

#map.drawcountries()



# We plot the locations of this bird trhougout the years

map.scatter(bird_df['location-long'].tolist(), bird_df['location-lat'].tolist(),

            latlon=True,c=bird_df['year'].tolist(),cmap='Paired')

map.colorbar()



plt.show()
# We get all travels from november (not only from the interesting birds)

november_df = df[df["month"] == "November"]



# We create the same map as before

map_november = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=68,llcrnrlon=-10,urcrnrlon=65,lat_ts=20,resolution='i')

map_november.drawcoastlines()

map_november.scatter(november_df['location-long'].tolist(), november_df['location-lat'].tolist(),

            latlon=True,c=november_df["year"].tolist(),cmap='Dark2',s=10)

map_november.colorbar()

plt.show()
# We get all travels from august (just to illustrate the difference)

august_df = df[df["month"] == "August"]



# We create the same map as before

map_august = Basemap(projection='merc',llcrnrlat=-5,urcrnrlat=68,llcrnrlon=-10,urcrnrlon=65,lat_ts=20,resolution='i')

map_august.drawcoastlines()

map_august.scatter(august_df['location-long'].tolist(), august_df['location-lat'].tolist(),

            latlon=True,c=august_df["year"].tolist(),cmap='Dark2',s=10)

map_august.colorbar()

plt.show()