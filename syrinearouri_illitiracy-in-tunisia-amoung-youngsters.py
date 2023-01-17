### Let's start by importing basic dependencies for data processing.


import numpy as np 
import pandas as pd 

# Checking the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import glob as glob
print(glob.glob("*.csv"))
# Importing essential files for map viewing

import pandas as pd
import geopandas as gpd
import geoviews as gv

gv.extension('bokeh')
geometries = gpd.read_file('../input/TUN_adm1.shp')
geometries.boundary
geometries.as_matrix   
idr = pd.read_csv('../input/codes_maps_tn.csv')
idr.columns
gdf = gpd.GeoDataFrame(pd.merge(geometries, idr))
gdf.head()
plot_opts = dict(tools=['hover'], width=550, height=800, color_index='Valeur',
                 colorbar=True, toolbar=None, xaxis=None, yaxis=None)
gv.Polygons(gdf, vdims=['Region', 'Valeur'], label='illiterates aged 10 and more by Delegation in Tunisia').opts(plot=plot_opts)
