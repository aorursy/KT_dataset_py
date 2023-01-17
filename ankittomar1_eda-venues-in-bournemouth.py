import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Load the dataset

df = pd.read_csv('../input/bournemouth_venues.csv')



# how big is the dataset is 

df.shape # just 100 rows 4 features 

# look for the data 

from pandas_profiling import ProfileReport

ProfileReport(df)
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



lat = df['Venue_Latitude']

lon = df['Venue_Longitude']



plt.figure(figsize=(8, 8))

m = Basemap(projection='ortho', resolution=None, lat_0=lat[0], lon_0=lon[0])

m.bluemarble(scale=0.5);