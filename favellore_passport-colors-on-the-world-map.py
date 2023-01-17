import pandas as pd

import numpy as np

import geopandas as gpd

import geoplot as gplt

import geoplot.crs as gcrs

from matplotlib.colors import LinearSegmentedColormap
# import shape files using geopandas

shp = gpd.read_file("../input/countries-shape-files/ne_10m_admin_0_countries.shp")

shp.head()
# import color file using pandas

df = pd.read_csv("../input/passport-colors/colors.csv", sep=",", index_col=0)

df.head()
# manually change country names which do not match between first and second datasets

shp.SOVEREIGNT[shp.SOVEREIGNT == "Russia"] = "Russian Federation"

shp.SOVEREIGNT[shp.SOVEREIGNT == "The Bahamas"] = "Bahamas"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Ivory Coast"] = "Cote d'Ivoire (Ivory Coast)"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Democratic Republic of the Congo"] = "Congo (Dem. Rep.)"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Republic of the Congo"] = "Congo"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Cabo Verde"] = "Cape Verde"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Czechia"] = "Czech Republic"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Myanmar"] = "Myanmar [Burma]"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Republic of Serbia"] = "Serbia"

shp.SOVEREIGNT[shp.SOVEREIGNT == "East Timor"] = "Timor-Leste"

shp.SOVEREIGNT[shp.SOVEREIGNT == "United Republic of Tanzania"] = "Tanzania"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Vatican"] = "Vatican City"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Saint Vincent and the Grenadines"] = "St. Vincent and the Grenadines"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Vietnam"] = "Viet Nam"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Swaziland"] = "Eswatini"

shp.SOVEREIGNT[shp.SOVEREIGNT == "Somaliland"] = "Somalia"
# create a dictionary {country: {R: value, G:value, B:value}} from the second dataset 

color_dictionary = dict(df.to_dict("index"))

#print(color_dictionary)



# using for-loop go through the list of countries in the first dataset

# if the country is in the color_dictionary from the second dataset, then add the color to the color_list

# if the country is not in the color_dictionary from the second dataset, then add the white color to the color_list

colors_list = []

coountries_not_in_list = []

for i in range(shp.shape[0]):

    if shp.SOVEREIGNT.iloc[i] in color_dictionary: 

        colors_list.append(tuple((color_dictionary[shp.SOVEREIGNT.iloc[i]]["R"],

                                  color_dictionary[shp.SOVEREIGNT.iloc[i]]["G"],

                                  color_dictionary[shp.SOVEREIGNT.iloc[i]]["B"])))

    else: 

        colors_list.append(tuple((1, 1, 1)))

        coountries_not_in_list.append(shp.SOVEREIGNT.iloc[i])

#print(colors_list)



# create custom colormap in matplotlib by passing color list to LinearSegmentedColormap

cmap_name = 'custom_colormap'

custom_colormap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=len(colors_list))
gplt.choropleth(shp, projection=gcrs.Robinson(), hue=shp.index, cmap=custom_colormap, k = None, figsize=(15, 7), linewidth=0,).set_global()

