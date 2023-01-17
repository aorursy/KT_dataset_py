print("Installing geopandas...")



# We need to install geopandas and descartes using PIP because they are 

# not installed on Jupyter by default. 



!pip install geopandas

!pip install descartes 
import geopandas as gpd

import pandas as pd

import matplotlib.pyplot as plt



# Next, print out the shapefile. 

# It won't look like much because we are looking at geographic coordinates. 



print("Loading Shapefile...")



# If using your files, replace below filename ("../input/region-boundaries-uk/NUTS_Level_1__January_2018__Boundaries.shp") with the 

# shapefile filename you uploaded. You need all the files though including the .dbf, shx. .... since they are all connected to the .shp file.

#../input/counties-and-unitary-authorities-england-shapefile

shapefile = gpd.read_file("../input/region-boundaries-uk/NUTS_Level_1__January_2018__Boundaries.shp")



# The "head" function prints out the first five rows in full, so you can see

# the columns in the data set too! 



shapefile.plot(figsize=(10, 10))
shapefile = shapefile.rename(index=str, columns={'nuts118nm': 'Region'}) # Renaming the column in the shape file to use 

shapefile.head(15)
#This cell imports the Stage 2 data as a Pandas dataframe

Aspira_data = pd.read_csv("../input/number-interested-in-aspira-per-region/num_interested_in_aspira.csv") #import the data

Aspira_data.head(20)
#Remove any spaces from the column headings

Aspira_data.columns = [col.strip() for col in Aspira_data.columns] 

merged = pd.merge(shapefile, Aspira_data, on='Region') 

print(merged.columns) #Print out the column headings.
# set a variable that will call whatever column we want to visualise on the map

variable = 'Num_interested'

# create figure and axes for Matplotlib

fig, ax = plt.subplots(figsize=(10, 6)) #Define the size of the figure

# create map

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')

# remove the axis

ax.axis("off")

# add a title

ax.set_title("University Neurodiversity Interest by Region", fontdict={"fontsize": "16", "fontweight" : "3"})

# create an annotation for the data source

ax.annotate("Source: Aspira Data Collection Survey, 2020",xy=(0.4, .08),  xycoords="figure fraction", horizontalalignment="left", verticalalignment="top", fontsize=10, color="#555555")



#Create Colour bar

vmin = merged[variable].min()

vmax = merged[variable].max()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range

sm._A = []

# add the colorbar to the figure

cbar = fig.colorbar(sm, orientation='vertical', shrink=0.8) # Shrink variable makes the colourbar slighly smaller
print(merged[variable])
fig.savefig('map_export.png', dpi=300)