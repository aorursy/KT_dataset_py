# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print("Installing geopandas...")



# We need to install geopandas and descartes using PIP because they are 

# not installed on Jupyter by default. 



!pip install geopandas

!pip install descartes 
import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd



# Next, print out the shapefile. 

# It won't look like much because we are looking at geographic coordinates. 



print("Loading Shapefile...")



# If using your files, replace below filename ("../input/region-boundaries-uk/NUTS_Level_1__January_2018__Boundaries.shp") with the 

# shapefile filename you uploaded. You need all the files though including the .dbf, shx. .... since they are all connected to the .shp file.

#../input/counties-and-unitary-authorities-england-shapefile

shapefile = gpd.read_file("../input/counties-and-unitary-authorities-england-shapefile/Counties_and_Unitary_Authorities__December_2016__Boundaries.shp")



# The "head" function prints out the first five rows in full, so you can see

# the columns in the data set too! 



shapefile.plot(figsize=(10, 10))