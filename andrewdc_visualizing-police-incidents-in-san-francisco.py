import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# handles geodata
import geopandas as gp
# converts coordinate pairs into points that can be interpreted by geopandas
from shapely.geometry import Point
# map plotting
import geoplot as gplt
import geoplot.crs as gcrs
# geoplot is based on cartopy
import cartopy
import cartopy.crs as ccrs
# reading the shapefile
sftracts = gp.read_file("../input/shapefiles-for-sf/sf_tracts_and_neighborhoods.dbf")

# we don't need these columns in this notebook
sftracts.drop(columns=["geoid", "shape_area", "shape_len"], inplace=True)

# setting the census tract ID as index will make map creation easier later on!
sftracts.set_index("tractce10", inplace=True)

sftracts.head()
sfnhoods = gp.read_file("../input/shapefiles-for-sf/sf_nhoods.dbf")
sfnhoods.set_index("nhood", inplace=True)

sfnhoods.head(5)
file = "../input/sf-police-calls-for-service-and-incidents/police-department-incidents.csv"

#read file without Location column
sf_incidents = pd.read_csv(file, usecols=lambda x: x not in ["Location"],
                           dtype={"IncidntNum":str})

# convert to datetime and merge date and time columns
sf_incidents["Date"] = pd.to_datetime(sf_incidents.Date) 
sf_incidents["Date"] = pd.to_datetime(sf_incidents.Date.dt.date.astype(str)
                                      + " " + sf_incidents.Time)
sf_incidents.drop("Time", axis=1, inplace=True)

# convert coords to points
sf_incidents["Coordinates"] = (sf_incidents[["X", "Y"]]
                               .apply(tuple, axis=1)
                               .apply(Point)
                              ) 
# convert dataframe to geodataframe
sf_incidents = gp.GeoDataFrame(sf_incidents, geometry="Coordinates")

sf_incidents.head()
# # Uncomment this when working locally!
# sf_incidents.crs = sftracts.crs # Making sure the map projections of both geodataframes are the same
# sf_incidents = gp.sjoin(sf_incidents, sftracts) # joining the geodataframes on its spatial geometries
# sf_incidents.rename(columns={"index_right":"tractce10"}, inplace=True)
# sf_incidents.iloc[:,::-1].head()
file2 = "../input/spatiallyjoineddata/police_incidents_after_spatial_join.csv"
sf_incidents = pd.read_csv(file2, parse_dates=[4], dtype={"tractce10":str})

# Restore the leading zeros in this column (optional)
sf_incidents["IncidntNum"] = sf_incidents.IncidntNum.apply(lambda x: "%09d" % x)

# Annoyingly we have to recreate the geometry from scratch
sf_incidents["Coordinates"] = sf_incidents[["X", "Y"]].apply(tuple, axis=1).apply(Point) 
sf_incidents = gp.GeoDataFrame(sf_incidents, geometry="Coordinates")

sf_incidents.iloc[:,::-1].head()
sf_incidents_sub = sf_incidents.loc[sf_incidents.Date.dt.year==2015]
(sf_incidents_sub.nhood.value_counts()
 .to_frame().rename(columns={"nhood":"Incidents"}).head(5))
(sf_incidents_sub.tractce10.value_counts()
 .to_frame().rename(columns={"tractce10":"Incidents"}).head(5))
drug_inc_by_tract = gp.GeoDataFrame((sf_incidents_sub
                                     .loc[sf_incidents_sub["Category"] == "DRUG/NARCOTIC"]
                                     .tractce10.value_counts()
                                     .to_frame()
                                     .rename(columns={"tractce10":"Incidents"})
                                    ).merge(sftracts.geometry.to_frame(),
                                            left_index=True, right_index=True)
                                   )
drug_inc_by_tract.head()
%config InlineBackend.figure_format = 'svg'
plt.style.use("seaborn-white")

fig = plt.figure(figsize=(12,10))
fig.suptitle("Various Map Projections", fontweight="bold", fontsize=20)

ax1 = plt.subplot(221, projection=ccrs.Mollweide())
ax1.coastlines()
ax1.stock_img()
ax1.set_anchor("N")
ax1.set_title("Mollweide Projection", fontweight="bold")

ax2=plt.subplot(222,projection=ccrs.Mercator())
ax2.coastlines()
ax2.stock_img()
ax2.set_anchor("N")
ax2.set_title("Mercator Projection", fontweight="bold")

ax3 = plt.subplot(223, projection=ccrs.InterruptedGoodeHomolosine())
ax3.coastlines()
ax3.stock_img()
ax3.set_anchor("N")
ax3.set_title("Goode Homolosine Projection", fontweight="bold")

ax4 = plt.subplot(224, projection=ccrs.AlbersEqualArea())
ax4.coastlines()
ax4.stock_img()
ax4.set_anchor("N")
ax4.set_title("Albers Equal Area Projection", fontweight="bold")
plt.show()
ax = gplt.polyplot(sftracts.geometry, projection = gcrs.LambertConformal(),
                   figsize=(12,8), edgecolor="k")
ax.set_title("Census Tract Boundaries in San Francisco", fontweight="bold", fontsize=16)
plt.show()
ax = gplt.choropleth(drug_inc_by_tract, projection=gcrs.LambertConformal(), hue="Incidents",
                     cmap="magma_r", k=None, linewidth=0, figsize=(10,8), legend=True)
gplt.polyplot(sftracts.geometry, projection = gcrs.LambertConformal(), edgecolor="k",
              linewidth=1, ax=ax)
ax.set_title("2015 SF Police Incidents by Census Tract\nCategory: Drug/Narcotic",
             fontweight="bold", fontsize=16)
plt.show()
ax = gplt.aggplot(sf_incidents_sub.loc[sf_incidents_sub["Category"] == "ASSAULT"],
                  projection=gcrs.LambertConformal(),
                  hue="Date", agg=len, by="tractce10",
                  geometry=sftracts, cmap="coolwarm", linewidth=0, figsize=(12,8),
                  vmin=0, vmax = 400)
gplt.polyplot(sftracts.geometry, projection=gcrs.LambertConformal(),
              linewidth=1, edgecolor="k", ax=ax)
ax.set_title("2015 SF Police Incidents by Census Tract\nCategory: Assault",
             fontweight="bold", fontsize=16)
plt.show()
ax = gplt.aggplot(sf_incidents_sub,
                  projection=gcrs.LambertConformal(),
                  hue="PdId", agg=lambda x: np.log10(len(x)), by="nhood",
                  geometry=sfnhoods.geometry, cmap="viridis", linewidth=0, figsize=(12,8),
                  vmin=2, vmax=4)
gplt.polyplot(sftracts.geometry, gcrs.LambertConformal(), ax=ax, linewidth=0.5, edgecolor="w")
gplt.polyplot(sfnhoods.geometry, gcrs.LambertConformal(), ax=ax, edgecolor="k")
ax.set_title("2015 SF Police Incidents (Total) by Neighborhood\n Log-Transformed",
             fontweight="bold", fontsize=16)
plt.show()
def crime_change(g, t0=2014, t1=2015): # calculates percentage changes
    g = g.dt.year
    a = len(g.loc[g==t1])
    b = len(g.loc[g==t0])
    if b > 0:
        c = 100*(a-b)/b
    elif ((b == 0) & (a > 0)):
        c = np.nan
    else: 
        c = 0
    return c

def crime_change_data(data, t0=2014, t1=2015, category=None, geography="nhood"):
    shapes = {"nhood": sfnhoods, "tractce10": sftracts}
    if category is None:
        g = data.groupby(geography).Date
    else:
        g = data.loc[data["Category"]==category].groupby(geography).Date
    return (gp.GeoDataFrame(g.apply(crime_change, t0=t0, t1=t1)
                           .to_frame().rename(columns={"Date":str(t0)+str(t1)+"change"})
                           )
            .join(shapes[geography], how="right")
           )
crime_change1617 = (crime_change_data(sf_incidents.loc[sf_incidents["nhood"] != "Presidio"],
                                     t0=2016, t1=2017, geography="tractce10", category=None)
                    .dropna())
ax = gplt.polyplot(sftracts.geometry, gcrs.LambertConformal(), facecolor="lightgrey",
                   figsize=(12,8), linewidth=0)
gplt.choropleth(crime_change1617, gcrs.LambertConformal(), hue="20162017change",
                ax=ax, k=None, cmap="RdBu_r", edgecolor="k", linewidth=0.5,
                vmin=-30, vmax=30, legend=True, legend_kwargs={"extend":"both"})
gplt.polyplot(sfnhoods.geometry, gcrs.LambertConformal(), ax=ax, edgecolor="k")
ax.set_title("% Change in SF Total Incidents by Census Tract\n2016 - 2017",
             fontweight="bold", fontsize=16)
plt.show()
