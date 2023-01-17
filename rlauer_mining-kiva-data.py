# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loansDF = pd.read_csv('../input/kiva_loans.csv')
loansDF.head()
loansDF.shape
loansDF[loansDF.drop('tags', axis=1).isnull().any(axis=1)]
#entries without region name:
loansDF[loansDF["region"].isnull()].shape
loansDF.drop(loansDF[loansDF["region"].isnull()].index).shape
mpiregionsDF = pd.read_csv("../input/kiva_mpi_region_locations.csv")
mpiregionsDF.sort_values("ISO").head()
mpiregionsDF[["country","region"]].describe()
#Let's look at nans:
mpiregionsDF[mpiregionsDF.isnull().any(axis=1)]
mpiClean = mpiregionsDF.drop(mpiregionsDF[mpiregionsDF[["lat","lon"]].isnull().any(axis=1)].index)
# check
mpiClean[mpiClean.isnull().any(axis=1)]
mpiClean.shape
mpiClean[["country","region"]].describe()
mpiClean.head()
#match geo data with country and region names
loansgeoDF = pd.merge(loansDF,mpiClean, on=["country","region"], how="inner", sort=False, validate="many_to_one")
#loansgeoDF.head()
print(loansgeoDF.shape)
loansgeoAll = pd.merge(loansDF,mpiClean, on=["country","region"], how="left", sort=False, validate="many_to_one")
print(loansgeoAll.shape)
#show those with lat, long = nan:
loansgeoAll[["country","region","lender_count"]][loansgeoAll[["lat","lon"]].isnull().any(axis=1)].groupby(["country","region"]).aggregate(sum)
from bokeh.io import output_file, show, output_notebook
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.tile_providers import STAMEN_TONER, STAMEN_TERRAIN_RETINA
from pyproj import Proj, transform
output_notebook()
bound = 20000000 #meters
fig = figure(tools='pan, wheel_zoom', x_range=(-bound, bound), y_range=(-bound/5, bound/5))
fig.axis.visible = False
fig.add_tile(STAMEN_TERRAIN_RETINA)

lon =  loansgeoDF["lon"].values
lat =  loansgeoDF["lat"].values
lenders = loansgeoDF["lender_count"]

from_proj = Proj(init="epsg:4326")
to_proj = Proj(init="epsg:3857")

x, y = transform(from_proj, to_proj, lon, lat)

fig.circle(x, y, size=lenders/20., alpha=0.6,
          color= "blue",   #;  {'field': 'rate', 'transform': color_mapper},
       ) #fill_alpha=0.7, line_color="white", line_width=0.5)

#output_file("stamen_toner_plot.html")
show(fig)


