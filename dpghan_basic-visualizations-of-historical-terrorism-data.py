from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")



terror = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1', 

                          usecols=[0, 1, 2, 3, 8, 9, 10, 13, 14, 28, 34, 

                          35, 83, 100, 103])



terror = terror.rename(columns={'eventid':'id', 'iyear':'year', 

                                          'imonth':'month', 'weaptype1':'weapon_type',

                                          'targtype1_txt':'target','iday':'day',

                                          'country_txt':'country','region_txt':'region_type',

                                          'region':'region_id','attacktype1':'attacktype_id',

                                          'targtype1':'target_id','targtype1_txt':'target',

                                          'nkill':'deaths','nwound':'injuries'})



terror['deaths'] = terror['deaths'].fillna(0).astype(int)

terror['injuries'] = terror['injuries'].fillna(0).astype(int)

terror['day'][terror.day == 0] = 1

terror['month'][terror.month == 0] = 1

terror['date'] = pd.to_datetime(terror[['day', 'month', 'year']])



terror.head()
corr = terror[['region_id','deaths','year','attacktype_id','target_id','injuries','weapon_type']].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(9, 6))

cmap = sns.diverging_palette(200, 1, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, square=True, linewidths=.5, annot=True, ax=ax)
count_year = terror.groupby(['year']).count()

death_year = terror.groupby(['year']).mean()



f1 = plt.figure()

ax1 = f1.add_subplot(211)

ax1.plot(count_year.index, count_year.deaths)

ax1.set(title='Total fatalities over time',xlabel='Year',ylabel='Fatalities')



f2 = plt.figure()

ax2 = f2.add_subplot(212)

ax2.plot(death_year.index, death_year.deaths)

ax2.set(title='Average fatalities per terrorist attack',xlabel='Year',ylabel='Fatalities')



plt.show()
my_map = Basemap(projection='robin', resolution='l', area_thresh=1000.0, lon_0=0)



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 9

fig_size[1] = 7



data_long = terror['longitude'].tolist()

data_lat = terror['latitude'].tolist()



my_map.drawcountries()

my_map.fillcontinents(color='#8FDA71')

my_map.drawmapboundary()



my_map.drawmeridians(np.arange(0, 360, 30))

my_map.drawparallels(np.arange(-90, 90, 30))



x, y =my_map(data_long, data_lat)

my_map.plot(x,y, 'ro', markersize=1)

plt.title('Terrorist activity around the globe, from 1970 to 2015')