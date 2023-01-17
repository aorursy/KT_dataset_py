# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

import matplotlib.animation as animation

from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



try:

    dataset = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')

    print("Data read succesfully.")

except:

    print("Error reading data.")
# Get a glance at the data.

dataset.head()
regions = list(set(dataset.region_txt))

colors = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange', 'brown',\

          'aqua', 'lightpink', 'purple', 'lightgray', 'navy']

print(regions)
plt.figure(figsize=(15,8))

m = Basemap(llcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')

m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='burlywood', lake_color='lightblue', zorder=1)



# Plots a point for each attack in the given region.

def pltpoints(region, color = None, label = None):

    # Get X and Y coordinates from dataset that belong to given region.

    x, y = m(

            list(dataset.longitude[dataset.region_txt == region].astype('float')),\

            list(dataset.latitude[dataset.region_txt == region].astype('float'))\

            )

    # Display point for each X and Y coordina

    points = m.plot(x, y, "o", markersize = 4, color = color, label = label, alpha = .5)

    

for i, region in enumerate(regions):

    pltpoints(region, color = colors[i], label = region)

        

plt.title("Global Terrorism from 1970")

plt.legend(loc = 'lower left', prop={'size': 11})



plt.show()
plt.figure(figsize=(15,8))

m = Basemap(projection='lcc' ,resolution='c', lat_0=50, lat_1=50, lon_0=20, width=5000000, height=3500000)

m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='burlywood', lake_color='lightblue', zorder=1)



def pltpoints(region, color = None, label = None):

    x, y = m(

            list(dataset.longitude[dataset.region_txt == region].astype('float')),\

            list(dataset.latitude[dataset.region_txt == region].astype('float'))\

            )

    points = m.plot(x, y, "o", markersize = 4, color = color, label = label, alpha = .5)

    



pltpoints('Western Europe', color = 'Purple', label = region)

pltpoints('Eastern Europe', color = 'Red', label = region)

        

plt.title("Global Terrorism from 1970")



plt.show()
years_distinct = dataset.groupby('iyear').count()

mean_per_year = dataset.groupby('iyear').mean()



fig = plt.figure(figsize = (10,8))



ax1 = fig.add_subplot(1, 2, 1)

ax1.set(title = 'Total acts of terrorism', ylabel = 'Act Count', xlabel = 'Year')

ax1.plot(years_distinct.index, years_distinct.eventid)



ax2 = fig.add_subplot(1, 2, 2)

ax2.set(title="Average number of deaths per act", ylabel = 'Average death count', xlabel = 'Year')

ax2.plot(years_distinct.index, mean_per_year.nkill)



fig.autofmt_xdate()
# result = dataset.approxdate[dataset.iday == 2].count()

# print(result)



count1 = dataset.region[dataset.region_txt == 'Western Europe'].count()

count2 = dataset.region[dataset.region_txt == 'Eastern Europe'].count()

print(count1)

print(count2)
from netCDF4 import Dataset, netcdftime, num2date

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime



import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap



# Terrorism dataset

t_set = pd.read_csv('globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')

print(t_set.shape)



# Weather dataset

w_set = Dataset('_grib2netcdf-atls09-a82bacafb5c306db76464bc7e824bb75-FCfyPw.nc')



w_lats = w_set.variables['latitude'][:]

w_lons = w_set.variables['longitude'][:]

w_time = w_set.variables['time'][:]

w_temp = w_set.variables['t2m'][:]

w_clouds = w_set.variables['tcc'][:]



time_units = w_set.variables['time'].units



try:

    time_cal = w_set.variables['time'].calendar

except AttributeError:

    time_cal = u"gregorian"

	

date_var = []

date_var.append(num2date(w_time, units = time_units, calendar = time_cal))



w_set.close() # Close weather set.





# Returns index of nearest lat value in weather coords.

# - Param target_val: the latitude value from terrorist db.

def t_lat_to_w_lat_index(target_val):

    return np.abs(w_lats - target_val).argmin()

	

# Returns index of nearest lon value in weather coords.

# - Param target_val: the longitude value from terrorist db.

def t_lon_to_w_lon_index(target_val):

    base_dif = 180 # Lons in weather data go from 0 - 360

                   # whereas in the terrorist db they go from -180 to 180

    return np.abs(w_lons - (target_val + base_dif)).argmin()



# Returns amount of steps since epoch time.

def days_from_epoch(year, month, day):

    epoch_datetime = datetime(2016, 1, 1) # First record in weather data.

    this_datetime = datetime(year, month, day)

    return (this_datetime - epoch_datetime).days



# Add a full date column.

def full_date(row):

    date_str = '%s/%s/%s' % (row['iyear'], row['imonth'], row['iday'])

    return date_str



# Connect a weather db column to 

def connect(row, w_col):

    if row.iyear == 2016: # Temp!

        epoch_time = days_from_epoch(row.iyear, row.imonth, row.iday)

        lat_index = t_lat_to_w_lat_index(row.latitude)

        lon_index = t_lon_to_w_lon_index(row.longitude)

        return w_col[epoch_time, lat_index, lon_index]

    return None



	

# print(t_set['latitude'].min(), t_set['latitude'].max())

# print(w_lats.min(), w_lats.max())



# print()

# print(t_set['longitude'].min(), t_set['longitude'].max())

# print(w_lons.min(), w_lons.max())





# Add date to terrorism rows.

# t_set['date'] = t_set.apply(lambda row: full_date(row), axis=1)



# Add temp data to terrorism rows.

t_set['temp'] = t_set.apply(lambda row: connect(row, w_temp), axis=1)

t_set['clouds'] = t_set.apply(lambda row: connect(row, w_clouds), axis=1)



print(t_set['clouds'][t_set.iyear == 2016])
