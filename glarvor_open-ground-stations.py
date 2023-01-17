import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
zone, year, param = 'NW', '2016', 'hu'

fname = '/kaggle/input/meteonet/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations_'+year+".csv"

df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
display(df.head())

display(df.tail())
date = '2016-01-01T06:00:00'

d_sub = df[df['date'] == date]



display(d_sub.head())

display(d_sub.tail())
d_sub['dd']
plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')

plt.show()
import cartopy.crs as ccrs

import cartopy.feature as cfeature
# Coordinates of studied area boundaries (in °N and °E)

lllat = 46.25  #lower left latitude

urlat = 51.896  #upper right latitude

lllon = -5.842  #lower left longitude

urlon = 2  #upper right longitude

extent = [lllon, urlon, lllat, urlat]



fig = plt.figure(figsize=(9,5))



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



# Plot the data

plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')  # Plot



# Add coastlines and borders

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



# Adjust the plot to the area we defined 

#/!\# this line causes a bug of the kaggle notebook and clears all the memory. That is why this line is commented and so

# the plot is not completely adjusted to the data

# Show only the area we defined

#ax.set_extent(extent)



plt.show()