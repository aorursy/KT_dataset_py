import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for graphing

from mpl_toolkits.basemap import Basemap



# Data types for each feature

ufo_data = pd.read_csv('../input/scrubbed.csv', usecols=[0, 1, 2, 9, 10], low_memory=False)

ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], errors='coerce')

ufo_data.insert(1, 'year', ufo_data['datetime'].dt.year)

ufo_data['year'] = ufo_data['year'].fillna(0).astype(int)

ufo_data['city'] = ufo_data['city'].str.title()

ufo_data['state'] = ufo_data['state'].str.upper()

ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'], errors='coerce')

ufo_data = ufo_data.rename(columns={'longitude ':'longitude'})



us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',

                        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',

                        'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',

                        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',

                        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])



# UFO sightings in United States only (70,805 rows)

ufo_data = ufo_data[ufo_data['state'].isin(us_states)].sort_values('year')

ufo_data = ufo_data[(ufo_data.latitude > 15) & (ufo_data.longitude < -65)]

ufo_data = ufo_data[(ufo_data.latitude > 50) & (ufo_data.longitude > -125) == False]

ufo_data = ufo_data[ufo_data['city'].str.contains('\(Canada\)|\(Mexico\)') == False]



# Create subsets for selected states

az_ufo_data = ufo_data[ufo_data['state'].str.contains('AZ') == True]

fl_ufo_data = ufo_data[ufo_data['state'].str.contains('FL') == True]

ny_ufo_data = ufo_data[ufo_data['state'].str.contains('NY') == True]

oh_ufo_data = ufo_data[ufo_data['state'].str.contains('OH') == True]
ufo_years = ufo_data[ufo_data.year != 0]



groupby_year = ufo_years['year'].groupby(ufo_years['year']).count().plot(kind='line')


az_ufo_years = az_ufo_data[az_ufo_data.year != 0]

fl_ufo_years = fl_ufo_data[fl_ufo_data.year != 0]

ny_ufo_years = ny_ufo_data[ny_ufo_data.year != 0]

oh_ufo_years = oh_ufo_data[oh_ufo_data.year != 0]





a = az_ufo_years['year'].groupby(az_ufo_years['year']).count()

b = fl_ufo_years['year'].groupby(fl_ufo_years['year']).count()

e = ny_ufo_years['year'].groupby(ny_ufo_years['year']).count()

f = oh_ufo_years['year'].groupby(oh_ufo_years['year']).count()



fig, ax = plt.subplots()

ax.plot(a, label='AZ')

ax.plot(b, label='FL')

ax.plot(e, label='NY')

ax.plot(f, label='OH')



legend = ax.legend(loc='upper left', shadow=True)
oh_ufo_data.describe()
oh_ufo_data[oh_ufo_data['longitude'] < -85]
oh_ufo_data = oh_ufo_data[oh_ufo_data.longitude != -106.280024]
oh_ufo_data.describe()
oh_ufo_years = oh_ufo_data[oh_ufo_data['year'] != 0]
oh_ufo_years.groupby(['year']).size().plot(kind='line');




plt.figure(figsize=(12,8))

OH = Basemap(projection='mill', llcrnrlat = 38, urcrnrlat = 42.5, llcrnrlon = -85.5, urcrnrlon = -80, 

             resolution = 'h')

OH.drawcoastlines()

OH.drawcountries()

OH.drawstates()

x, y = OH(list(oh_ufo_data["longitude"].astype("float")), list(oh_ufo_data["latitude"].astype(float)))

OH.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")



plt.title('UFO Sightings in Ohio')

plt.show()
az_ufo_data.describe()
az_ufo_data[az_ufo_data['longitude'] < -115]
az_ufo_data = az_ufo_data[az_ufo_data.longitude > -115]
az_ufo_data[az_ufo_data['longitude'] > -109]
az_ufo_data = az_ufo_data[az_ufo_data.longitude < -109]
az_ufo_years = az_ufo_data[az_ufo_data['year'] != 0]
az_ufo_years.groupby(['year']).size().plot(kind='line');
plt.figure(figsize=(12,8))

AZ = Basemap(projection='mill', llcrnrlat = 31, urcrnrlat = 38, 

             llcrnrlon = -115.5, urcrnrlon = -108, 

             resolution = 'h')

AZ.drawcoastlines()

AZ.drawcountries()

AZ.drawstates()

x, y = AZ(list(az_ufo_data["longitude"].astype("float")), list(az_ufo_data["latitude"].astype(float)))

AZ.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")



plt.title('UFO Sightings in Arizona')

plt.show()
fl_ufo_years = fl_ufo_data[fl_ufo_data.year != 0]



fl_ufo_years['year'].groupby(fl_ufo_years['year']).count().plot(kind='line');
fl_ufo_data.describe()
fl_ufo_data[fl_ufo_data['latitude'] > 31]
fl_ufo_data = fl_ufo_data[fl_ufo_data.latitude < 31]
fl_ufo_data.describe()
plt.figure(figsize=(12,8))

FL = Basemap(projection='mill', llcrnrlat = 24, urcrnrlat = 31.5, 

             llcrnrlon = -88, urcrnrlon = -79.5, 

             resolution = 'h')

FL.drawcoastlines()

FL.drawcountries()

FL.drawstates()

x, y = FL(list(fl_ufo_data["longitude"].astype("float")), list(fl_ufo_data["latitude"].astype(float)))

FL.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")



plt.title('UFO Sightings in Florida')

plt.show()
ny_ufo_years = ny_ufo_data[ny_ufo_data.year != 0]



ny_ufo_years['year'].groupby(ny_ufo_years['year']).count().plot(kind='line');
ny_ufo_data.describe()
plt.figure(figsize=(11,7))

NY = Basemap(projection='mill', llcrnrlat = 40, urcrnrlat = 45.25, 

             llcrnrlon = -80, urcrnrlon = -71.75, 

             resolution = 'h')

NY.drawcoastlines()

NY.drawcountries()

NY.drawstates()

x, y = NY(list(ny_ufo_data["longitude"].astype("float")), list(ny_ufo_data["latitude"].astype(float)))

NY.plot(x, y, "go", markersize = 4, alpha = 0.8, color = "green")



plt.title('UFO Sightings in New York')

plt.show()