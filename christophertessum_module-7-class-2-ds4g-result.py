# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import xarray

import re

from datetime import datetime



import holoviews as hv

hv.extension('matplotlib')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files = sorted(glob.glob("/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_201807*.tif"))
ds = xarray.concat([xarray.open_rasterio(f) for f in files], dim="time")



ds
ds.name = "no2"

times = []

for f in sorted(glob.glob("/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_201807*.tif")):

    t = re.search("([0-9]{8}T[0-9]{6})", f)[0]

    times.append(datetime.strptime(t, "%Y%m%dT%H%M%S"))



ds.coords["time"]= times
ds
ds.isel(time=0, band=0).plot(size=6);
hds = hv.Dataset(ds.isel(band=0), name='testdata').to(hv.Image, kdims=['x', 'y']).options(fig_inches=(10, 3), colorbar=True, cmap='viridis')



hds
print(ds.isel(band=0).mean().values) # This selects by index

print(ds.sel(band=1).mean().values) # This selects by value
single_loc = ds.sel(band=1, y=18.2, x=-66.25, method='nearest')



single_loc
single_loc.plot(size=6);
single_loc.mean().values