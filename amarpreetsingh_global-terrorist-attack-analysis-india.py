import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# For plotting on map

from mpl_toolkits.basemap import Basemap

from matplotlib import cm
df = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding = "ISO-8859-1",low_memory=False)
df.head()

#Year wise all terrorrist attacks

year_by_attacks = df[['iyear','eventid']].groupby('iyear').count()

year_by_attacks.plot(legend=False,title='Year wise all terrorrist attacks',figsize=(10, 5))

plt.xlabel('Year')

plt.ylabel('Attacks')

plt.show()
# Country wise terrorist attacks (top 20)

country_by_attacks = df[['country_txt','eventid']].groupby('country_txt').count().sort_values('eventid',ascending=0).head(20)

country_by_attacks.plot(kind='bar',legend=False)

plt.xlabel('Country')

plt.ylabel('Attacks')

plt.show()
# Attacks in India by state(top 15)

india_attacks = df[df['country_txt']=='India']

india_attack_by_state = india_attacks[['provstate','eventid']].groupby('provstate').count().sort_values('eventid',ascending=0).head(15)

india_attack_by_state.plot(kind='bar',legend=False)

plt.xlabel('Country')

plt.ylabel('No of Attacks')

plt.show()
#State wise casualities based on target type (Top 10)

india_atacks_by_city = india_attacks[india_attacks['iyear'] > 2006].groupby(['provstate','targtype1_txt']).count()

india_atacks_by_city = (india_atacks_by_city.reset_index()).sort_values('nkill',ascending=0)

india_atacks_by_city = india_atacks_by_city[['provstate','nkill','targtype1_txt']].head(10)

india_atacks_by_city
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(india_atacks_by_city['provstate'],india_atacks_by_city['targtype1_txt'], s=india_atacks_by_city['nkill']) # Added third variable income as size of the bubble

plt.show()
#Year Wise Attacks stacked on weapons used

year_wise_weapon_type = india_attacks.groupby(['iyear','weaptype1_txt']).count()

year_wise_weapon_type_plot = year_wise_weapon_type[['eventid']].unstack().plot(kind='bar',stacked=True,figsize=(10, 5),title="Year wise Attacks (per weapon)")

year_wise_weapon_type_plot.set_xlabel("Years")

year_wise_weapon_type_plot.set_ylabel("No of Attacks")

h, l = year_wise_weapon_type_plot.get_legend_handles_labels()

l = [x.split('eventid,')[1].split(')')[0] for x in l]

year_wise_weapon_type_plot.legend(l)

plt.show()
plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()

india_attacks_map = india_attacks[['latitude','longitude','nkill','nwound']].dropna()

lg=list(india_attacks_map['longitude'])

lt=list(india_attacks_map['latitude'])

casualities = [sum(x) for x in zip(india_attacks_map['nwound'], india_attacks_map['nkill'])]

x, y = map(lg, lt)

plt.scatter(x, y, s=casualities, marker="o", c=casualities, cmap=cm.Dark2, alpha=0.7)

plt.show()