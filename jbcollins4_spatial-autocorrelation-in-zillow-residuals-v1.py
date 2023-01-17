import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline
import pyproj

center_lon = -118.5  # A reasonable value for the Los Angeles area.

ps0 = '+proj=tmerc +lat_0=0.0 +lon_0=%.1f +y_0=0 +x_00 +k_0=0.9996 +units=m +ellps=WGS84' % center_lon

remap = pyproj.Proj(ps0)
props = pd.read_csv('../input/properties_2016.csv')
# Build a new dataframe that just includes the lon and lat, indexed by parcel ID.

lon = props['longitude'] / 1.0e6

lat = props['latitude'] / 1.0e6

parcelid = props['parcelid']

parcel_coords = pd.DataFrame({'parcelid': parcelid, 'lon': lon, 'lat': lat})

parcel_coords = parcel_coords.set_index('parcelid').dropna(axis=0)

parcel_coords.head()
# Make a plot to get a sense of the overall spatial distribution of the data we're 

# dealing with. Here we're just plotting a random sample of the parcel lon/lat coordinates.

dfs = parcel_coords.sample(n=5000)

plt.figure(figsize=(12, 12))

plt.plot(dfs['lon'], dfs['lat'], '.')
lon_min = -118.70

lon_max = -118.25

lat_min = 34.10

lat_max = 34.35
import csv

parcel_list = []

with open('../input/train_2016_v2.csv') as source:



    reader = csv.DictReader(source, delimiter=',')

    k = 0

    for rec in reader:

    

#         k += 1

#         if k % 10000 == 0:

#             print('Handling record %d' % k)

        

        pid = int(rec['parcelid'])

        lon = parcel_coords.loc[pid]['lon']

        lat = parcel_coords.loc[pid]['lat']

        

        # Only keep records that fall within our area of interest.

        if lon_min < lon < lon_max and lat_min < lat < lat_max:

            

            # Also, subsample the records a bit. A full sample is more than we need to make 

            # our point, and just slows things down.

            if np.random.random() < 0.1:

                

                # Get the projected coordinates.

                (xx, yy) = remap(lon, lat)

                

                # Add the relevant information to the parcel list.

                parcel_list.append({'xx': xx, 'yy': yy, 'resid': float(rec['logerror'])})

            

print('Keeping information for %d parcels' % len(parcel_list))
# Take another look at the spatial distribution of the parcels. This time we plot them

# in projected (x, y) coordinates.

df = pd.DataFrame.from_dict(parcel_list)

df.head()

plt.figure(figsize=(10, 10))

plt.plot(df['xx'], df['yy'], '.')
range_parameter = 300.0  # meters

distance = np.arange(0.0, 2000.0, 10.0)

spatial_similarity = np.exp(-distance / range_parameter)



plt.figure(figsize=(8, 4))

plt.plot(distance, spatial_similarity, '-')

plt.xlabel('parcel-to-parcel distance [meters]')

plt.ylabel('spatial similarity')
parcel_count = len(parcel_list)

distance_threshold = 1500.0  # meters

range_parameter = 300.0  # meters 

spatial_similarity = []

value_similarity = []

sample_count = 0



for ii in range(parcel_count):

    

#     if ii % 100 == 0:

#         print('%d / %d' % (ii, parcel_count))

        

    for jj in range(ii+1, parcel_count):

        dx = parcel_list[ii]['xx'] - parcel_list[jj]['xx']

        dy = parcel_list[ii]['yy'] - parcel_list[jj]['yy']

        dd = np.sqrt(dx**2 + dy **2)

        if dd < distance_threshold:

            dv = np.abs(parcel_list[ii]['resid'] - parcel_list[jj]['resid'])

            value_similarity.append(np.exp(-dv / 0.5))

            spatial_similarity.append(np.exp(-dd / range_parameter))

            sample_count += 1
plt.figure(figsize=(8,6))

plt.plot(spatial_similarity, value_similarity, '.')

plt.xlabel('spatial similarity')

plt.ylabel('value similarity')

plt.ylim(0, 1.02)
gamma = np.dot(spatial_similarity, value_similarity)

print('gamma = %f' % gamma)
nnn = 100

gamma_null = np.zeros(nnn)

for k in range(nnn):

    for i in range(len(spatial_similarity)):

        zz = np.random.choice(len(parcel_list), 2, replace=False)

        v0 = parcel_list[zz[0]]['resid']

        v1 = parcel_list[zz[1]]['resid']

        dv = np.abs(v0 - v1)

        value_similarity = np.exp(-dv /0.5)

        gamma_null[k] += value_similarity * spatial_similarity[i]

    print('%d / %d: null gamma: %.1f [vs. observed gamma %.1f]' % (k, nnn, gamma_null[k], gamma))
# Plot the distribution of the gamma values that we woudl get under the null 

# hypothesis of no spatial autocorrelation.

plt.figure(figsize=(10, 10))

plt.hist(gamma_null, bins=30)

plt.xlabel('Gamma Value')

plt.ylabel('Relative Frequency')

plt.title('Gamma Values In Absence Of Autocorrelation (Versus Observed Value %.1f)' % gamma)