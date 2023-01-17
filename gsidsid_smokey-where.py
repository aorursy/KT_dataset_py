import datetime as dt

import sqlite3

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
conn = sqlite3.connect('../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite')

c = conn.cursor()



# Fetch CA state code data using a simple WHERE query



c.execute("""SELECT FIRE_Size, FIRE_SIZE_CLASS, DISCOVERY_DATE, 

DISCOVERY_TIME, CONT_DATE, CONT_TIME, STAT_CAUSE_DESCR, FIPS_NAME,

LATITUDE, LONGITUDE

from Fires

where STATE = 'CA'""")
for row in c.fetchone():

    print(row)
# Data preparation (once)



column_names = ['fire_size', 'fire_size_class', 

                'discovery_date', 'discovery_time', 

                'cont_date', 'cont_time', 

                'stat_cause_descr', 'county_name', 

                'latitude', 'longitude']



raw_data = c.fetchall()

data_ar = np.array(raw_data) #turns list into array

df = pd.DataFrame(data_ar, columns = [column_names]) # turns array into dataframe

df.columns = df.columns.get_level_values(0) # dataframe has dumb multiindex issue, this fetches index lvl 1 and sets 
df['disc_clean_date'] = pd.to_datetime(df['discovery_date'] - pd.Timestamp(0).to_julian_date(), unit='D')

df['cont_clean_date'] = pd.to_datetime(df['cont_date'] - pd.Timestamp(0).to_julian_date(), unit='D')

df['discovery_month'] = df['disc_clean_date'].dt.strftime('%b')

df['discovery_weekday'] = df['disc_clean_date'].dt.strftime('%a')

df['disc_date_final'] = pd.to_datetime(df.disc_clean_date.astype('str') + ' ' + df.discovery_time, errors='coerce')

df['cont_date_final'] = pd.to_datetime(df.cont_clean_date.astype('str') + ' ' + df.cont_time, errors='coerce')

df['putout_time'] = df['cont_clean_date']-df['disc_clean_date']
df
from matplotlib.pyplot import figure

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')



axs = sns.countplot(x="stat_cause_descr", data=df)

axs.set_title("Causes of fires in CA")
import datetime



from matplotlib.pyplot import figure

figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')



# plot non-parametric kde on numeric datenum

df['ordinal'] = [x.toordinal() for x in df.disc_clean_date]

ax = df['ordinal'].plot(kind='kde', xlim=(min(df.disc_clean_date),max(df.disc_clean_date)))

# rename the xticks with labels

x_ticks = ax.get_xticks()

ax.set_xticks(x_ticks[::1])

xlabels = [datetime.datetime.fromordinal(int(x)).strftime('%Y-%m-%d') for x in x_ticks[::1]]

ax.set_xticklabels(xlabels);

ax.set_title("Fire observations over time")
df.loc[df.fire_size_class == "A", "fire_mag"] = 1/100

df.loc[df.fire_size_class == "B", "fire_mag"] = 10/100

df.loc[df.fire_size_class == "C", "fire_mag"] = 100/100

df.loc[df.fire_size_class == "D", "fire_mag"] = 300/100

df.loc[df.fire_size_class == "E", "fire_mag"] = 1000/100

df.loc[df.fire_size_class == "F", "fire_mag"] = 5000/100

df.loc[df.fire_size_class == "G", "fire_mag"] = 10000/100

df
from mpl_toolkits.basemap import Basemap



fig = plt.figure(figsize=(12, 12))

m = Basemap(projection='lcc', resolution='l', 

            lat_0=37.5, lon_0=-119,

            width=1E6, height=1.2E6)

#m.shadedrelief()

m.drawcoastlines(color='gray')

m.drawcountries(color='gray')

m.drawstates(color='gray')



cvals = []

for x in df.putout_time:

    if x is not pd.NaT and x.days < 30:

        cvals.append(int(x.days))

    elif x is not pd.NaT:

        cvals.append(30)

    else:

        cvals.append(0)



m.scatter(df['longitude'].values, df['latitude'].values, latlon=True,

          s=list(df['fire_mag'].values),

          c=cvals,

          cmap='Reds', alpha=0.6)



cbar = plt.colorbar(label=r'${duration}_{days}$')

#cbar.ax.set_yticklabels()
"""



For each fire observance take the month and county, get weather stats from NOAA (ftp://ftpcimis.water.ca.gov/pub2/annualMetric/) and append current conditions to row.



Elsewhere, for each county, get the average lat/long and create an artificial weight between 0 and 1 based on (fires observed in county / average fires observed in county)



Make a model that takes in month and county weather data and outputs a probability of a fire in that county. Multiply with the expected probability of a fire in the county and create a heatmap of probabilities of fires by county



Write a lot



"""