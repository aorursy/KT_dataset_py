import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') #ignore warning messages 
# Read data
data = pd.read_csv('../input/meteorite-landings.csv')
data.head()
# Are there missing values?
print(data.isnull().values.any())
print('Missing values for each column:\n', data.isnull().sum())
print('Total number of missing values: ', data.isnull().sum().sum())
# Drop data rows with NA values
valid = data.copy()
#valid.where(valid['nametype']=='Valid', inplace=True)
valid.dropna(inplace=True)

# Drop useless columns
valid.drop('GeoLocation', axis=1, inplace=True)
valid.drop('id', axis=1, inplace=True)

#Filter out weird years
valid = valid[valid['year']<2016]
valid = valid[valid['year']>860]

# Separate data where the meteorite has been observed ('Fell') and was only found ('Found')
fell = valid.where(valid['fall']=='Fell')
found = valid.where(valid['fall']=='Found')
valid.head()

# Are there missing values?
print(valid.isnull().values.any())
#print('Missing values for each column:\n', valid.isnull().sum())
#print('Total number of missing values: ', valid.isnull().sum().sum())
timerange = valid['year'].max()-valid['year'].min()
# Stacked Histogram for observed and found meteorites
#pd.DataFrame({'Found': found['year'], 'Observed': fell['year']}).plot.hist(stacked=True, bins=int(timerange/5))

# Histogram for total number of meteorites
valid.hist(column='year', bins=int(timerange/1), xlabelsize=12, ylabelsize=12, figsize=(9,6), \
            label='# meteorites per year', color='k')
plt.legend(prop={'size':15})

# Histogram for total number of meteorites only after 1900
Twentiethcentury = valid[valid['year']>1900]
timerange = Twentiethcentury['year'].max()-Twentiethcentury['year'].min()
Twentiethcentury.hist(column='year', bins=int(timerange/1), xlabelsize=12, ylabelsize=12, \
                               figsize=(9,6), label='# meteorites per year', color='k')
plt.legend(loc=2, prop={'size':15})
# Throw away all meteorites with masses too large
cutoff = 200
mass = valid[valid['mass']<cutoff]
#ObsMass = fell[fell['mass']<cutoff]
ObsMass = fell[fell['mass']<2e4]
# Compute the median and quantile of masses
median = valid['mass'].median()
q1 = valid['mass'].quantile(0.25)
q2 = valid['mass'].quantile(0.75)
ObsMedian = fell['mass'].median()
Obsq1 = fell['mass'].quantile(0.25)
Obsq2 = fell['mass'].quantile(0.75)

# Create Histogram for all meteorites
mass.hist(column='mass', bins=cutoff, xlabelsize=12, ylabelsize=12, figsize=(9,6), \
            label='mass in gram, binsize=1', color='k')
plt.axvline(median, label='median={0:.1f}g'.format(median), color='g')
plt.axvline(q1, label='25% quantile={0:.1f}g'.format(q1), color='r')
plt.axvline(q2, label='75% quantile={0:.1f}g'.format(q2), color='b')
plt.legend(loc=9, prop={'size':15})
plt.title('mass histogram of all meteorites', fontsize=20)

# Create Histogram only for observed meteorites
ObsMass.hist(column='mass', bins=200, xlabelsize=12, ylabelsize=12, figsize=(9,6), \
            label='mass in gram, binsize=100g', color='k')
plt.axvline(ObsMedian, label='median={0:.1f}kg'.format(ObsMedian/1000), color='g')
plt.axvline(Obsq1, label='25% quantile={0:.1f}kg'.format(Obsq1/1000), color='r')
plt.axvline(Obsq2, label='75% quantile={0:.1f}kg'.format(Obsq2/1000), color='b')
plt.legend(loc=1, prop={'size':15})
plt.title('Mass histogram of observed meteorites', fontsize=20)
types = valid['fall'].value_counts()
types.plot.pie(autopct='%.1f', fontsize=15, title='Percentage of observed and found meteorites')
# Pie Chart for occurences of different meteorite types
valid.head()
classes = valid['recclass'].value_counts()
classes[classes>100].plot.pie(autopct='%0.1f')
# Make the background map
plt.figure(figsize=(24,18))
m=Basemap(llcrnrlon=-180, llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")

m.plot(fell['reclong'], fell['reclat'], linestyle='none', marker='o', ms=1, color='red', label='observed')
m.plot(found['reclong'], found['reclat'], linestyle='none', marker='o', ms=1, color='black', label='found')
plt.legend(loc=6, prop={'size': 30}, markerscale=8)
plt.title('Geographic Distribution of meteorites', fontsize=25)
# Heaviest observed meteorites
heavy_obs = fell.sort_values('mass', ascending=False)
heavy_obs.head()
# Heaviest found meteorites
heavy = found.sort_values('mass', ascending=False)
heavy.head()
# Make the background map
plt.figure(figsize=(24,18))
m=Basemap(llcrnrlon=-180, llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")

maxSize=valid['mass'].max()
m.scatter(fell['reclong'], fell['reclat'], marker='o', s=fell['mass']/maxSize*10000, alpha=0.4, color='red', label='observed')
m.scatter(found['reclong'], found['reclat'], marker='o', s=found['mass']/maxSize*10000, alpha=0.4, color='black', label='found')
plt.legend(loc=6, prop={'size': 30}, markerscale=0.5)
plt.text(-170,-20, 'linear mass scale', fontsize=25)
plt.title('Bubble map with mass of meteorite encoded in size of bubble', fontsize=25)
