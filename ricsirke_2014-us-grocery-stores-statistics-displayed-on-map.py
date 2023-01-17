import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import os
print(os.listdir("../input/us-state-coordinates"))
print("../input")
for e in os.listdir("../input"):
    print(e)
usshapefilepath = "../input/us-states-cartographic-boundary-shapefiles/cb_2016_us_state_500k"
df_coordmap = pd.read_csv("../input/us-state-coordinates/state_coord_map.csv")[["code", "Latitude", "Longitude"]]
df_vars = pd.read_excel("../input/food-environment-atlas/DataDownload.xls", sheet_name="Variable List")
df_vars[df_vars["Category Code"] == "STORES" ].head()
df = pd.read_excel("../input/food-environment-atlas/DataDownload.xls", sheet_name="STORES")
df.head()
df_grocnum_state = df[["State", "GROC09", "GROC14"]].groupby("State").sum().reset_index()
df_grocnum_state = pd.merge(df_grocnum_state, df_coordmap, left_on="State", right_on="code")
df_grocnum_state = df_grocnum_state.drop('code', axis=1)
df_grocnum_state.sort_values('GROC14', ascending=False).head()
fig, ax = plt.subplots(figsize=(12,12))
m = Basemap(llcrnrlon=-128.217356,llcrnrlat=23.563639,
            urcrnrlon=-65.419440,urcrnrlat=51.212921,
              projection='cyl')
m.readshapefile(usshapefilepath, name='states', drawbounds=True)
m.drawmapboundary(fill_color='lightblue')
m.drawcountries()
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
m.drawcoastlines()

maxgroc = df_grocnum_state["GROC14"].max()
for index, row in df_grocnum_state.iterrows():
    markersize = int(math.floor(50*math.sqrt(row["GROC14"]/maxgroc)))
    m.plot(row["Longitude"], row["Latitude"], "o", markersize=markersize)
plt.figtext(.5,.72,"Comparison of the number of shops ", fontsize=18, ha="center")
plt.figtext(.5,.69,"the area of a circle is proportional to the number of shops in the given state", fontsize=14, ha="center")
#plt.title("")
plt.show()
