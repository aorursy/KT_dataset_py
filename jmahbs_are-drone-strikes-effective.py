import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from collections import Counter

from mpl_toolkits.basemap import Basemap
df=pd.read_csv("../input/PakistanDroneAttacksWithTemp Ver 11 (November 30 2017).csv", encoding = "ISO-8859-1")

# Open first and last rows

df.head(2)
df.tail(2)
numerical_cols = ['No of Strike',

       'Al-Qaeda', 'Taliban', 'Civilians Min', 'Civilians Max',

       'Foreigners Min', 'Foreigners Max', 'Total Died Min', 'Total Died Max',

       'Injured Min', 'Injured Max']
# Initial clean up steps

df_clean = df.copy()



df_clean.loc[:,numerical_cols].fillna(0, inplace=True) #replace NaNs with 0

#drop last row that sums values

try: df_clean.drop(df.index[-1], inplace=True) 

except: print('last row already dropped')



df_clean['Date_clean'] = pd.to_datetime(df_clean['Date'], errors='coerce')

# df_clean['Date_clean'].value_counts()
# Summarise distribution of values

df_clean.describe()
# convert negative deaths to positive 

df_clean['Civilians Min'] = df['Civilians Min'].abs()

df_clean['Civilians Max'] = df['Civilians Max'].abs()



# create average civilian death and group militants together

df_clean['Civilians Avg']= df_clean[['Civilians Min','Civilians Max']].mean(axis=1)

df_clean['Militants']= df_clean[['Al-Qaeda','Taliban']].sum(axis=1)
# Create dict, and then dataframe, with row count by city

city_dict = Counter(df.loc[df['City'].notnull(), 'City'])



# plot row count by city

fig, ax = plt.subplots()

ax.bar(city_dict.keys(), city_dict.values())



ax.set_xlabel('City')

ax.set_ylabel('Count')

ax.set_title('Cities by row frequency')



plt.xticks(rotation=90)



plt.show()
# clean up city names

df_clean['City'].replace('Hungu', 'Hangu', inplace=True)

df_clean['City'].replace('Khyber', 'Khyber Agency', inplace=True)

df_clean['City'].replace(['south waziristan','south Waziristan','South waziristan'], 'South Waziristan', inplace=True)
# create map figure

fig, ax = plt.subplots(figsize=(20, 16))

world_map = Basemap(projection='cyl', resolution=None)

                    #llcrnrlat=-90, urcrnrlat=90,

                    #llcrnrlon=-180, urcrnrlon=180)

world_map.bluemarble(scale=0.5)



#scatter drone strikes

world_map.scatter(df['Longitude'].values, df['Latitude'].values, latlon=True,

                  s=50, c='red', alpha=0.5)

# title

plt.title('Drone strikes (2004-2017) by location')



# legend

plt.scatter([], [], c='red', alpha=0.5, s=50,

                label='Drone strike')

plt.legend(scatterpoints=1, frameon=False,

                labelspacing=1, loc='lower left');



plt.show()
# I swap lat / lon pairs where lat is greater than 40 (based on max latitude in Pakistan)

# see https://en.wikipedia.org/wiki/Extreme_points_of_Pakistan, 



df_clean['Latitude_new'] = np.where(df_clean['Latitude']>40, df_clean['Longitude'], df_clean['Latitude'])

df_clean['Longitude_new'] = np.where(df_clean['Latitude']>40, df_clean['Latitude'], df_clean['Longitude'])
# As there are only a limited set of lat/lon pairs, I group the dataset together by lat/lon pairs

# to allow us to tie other values to lat/lon pairs

map_df = df_clean.groupby(['Latitude_new', 'Longitude_new']).sum().reset_index()

# map_df.drop(map_df.iloc[:,13:17], axis=1, inplace=True) # remove columns that become meaningless when summed

map_df.sort_values('No of Strike', axis=0, inplace=True) # sort by 'No of strike' so highest volume location on top
# create the map figure to plot

fig = plt.subplots(figsize=(14, 14))

pk_map = Basemap(projection='lcc',

              resolution='l',

              lat_0=30, lon_0=70,

              llcrnrlon=60, llcrnrlat=23,

              urcrnrlon=79, urcrnrlat=37)

pk_map.shadedrelief()

pk_map.drawcoastlines(color='blue')

pk_map.drawcountries(color='k', linewidth=1.5)



# scatter drone strikes

pk_map.scatter(map_df['Longitude_new'].values, map_df['Latitude_new'].values, latlon=True,

                s=200,

                c=map_df['No of Strike'].values, edgecolor='k',

                cmap='cool', alpha=0.6, zorder=2)



# Plot some relevant cities - source: google

city_labels = ['  Hyderabad','  Kabul', '  Islamabad', '  Karachi']

city_lonlat = [(68.35, 25.39),(69.20, 34.55), (73.04, 33.68), (67,24.86)]



for (lon, lat), label in zip(city_lonlat, city_labels):

    (lon, lat) = pk_map(lon, lat)

    plt.plot(lon, lat, 'ok', markersize=8)

    plt.text(lon, lat, label, fontsize=15, zorder=1)



# plot legend sidebar

plt.colorbar(fraction=0.039, pad=0.04, label=r'Number of strikes (2004-2016)')



# title

plt.title('Drone strike locations by strike volume (2004-2017)', fontsize=20)



plt.show()
# create dataframe with data summed by year to plot

year_df = df_clean.groupby(df_clean['Date_clean'].dt.year)['No of Strike', 'Militants', 'Civilians Avg'].sum().reset_index()



# label dataframe by which president in office for each year

year_df['President'] = year_df['Date_clean'].apply(lambda x: 'Bush' if x < 2009

                                                             else ('Trump' if x > 2016 else 'Obama'))



fig = sns.factorplot(x="Date_clean",y="No of Strike", hue="President", size=6, aspect=2, data=year_df)

plt.xlabel('Year')

plt.ylabel('Number of Drone Strikes')

plt.title("Annual Drone Strikes by US President", fontsize=20)



plt.show()
# Set up dataframe with civilian and militant deaths, manually entered

# source: https://en.wikipedia.org/wiki/Civilian_casualty_ratio for civilian and non-US Militant deaths

#         wikipedia pages for different conflicts for US Militant Deaths



comp_df = pd.DataFrame(columns=['Conflict Name',

                                'Civilian Deaths',

                                'Non-USA Militant Deaths',

                                'USA Militant Deaths'], 

                       data=[

                            ['USA-Korean War \n (1950s)', 2730000, 747000, 40000],

                            ['USA-Vietnam War \n (1960s)', 2000000, 1100000, 58220],

                            ['USA-Afghanistan War \n (2001-2015)', 26000, 66000, 2271],

                            ['USA-Iraq War \n (2003-2013)', 134100, 39900, 4497],

                            ['USA-Pakistan Drone Strikes \n (2004-2017)', df_clean['Civilians Avg'].sum(), df_clean['Militants'].sum(axis=0),0]

                            ])



# create columns to use when plotting as a chart

comp_df['Non-USA Deaths'] = comp_df[['Civilian Deaths', 'Non-USA Militant Deaths']].sum(axis=1)

comp_df['Non-USA Civilian%'] = comp_df['Civilian Deaths']/comp_df['Non-USA Deaths']

comp_df['Non-USA Militant%'] = comp_df['Non-USA Militant Deaths']/comp_df['Non-USA Deaths']



# plot conflicts together as stacked bar chart

fig = plt.figure(figsize=(14, 6))

p1 = plt.bar(comp_df.index, comp_df['Non-USA Civilian%'])

p2 = plt.bar(comp_df.index, comp_df['Non-USA Militant%'],

             bottom=comp_df['Non-USA Civilian%'],

             color='0.75')



plt.title('Civilian Death Ratio by Conflict (Excluding USA Militant Deaths)', fontsize=20)

plt.ylabel('% Percentage of Non-USA Deaths')

plt.xticks(comp_df.index, comp_df['Conflict Name'])

plt.legend((p1[0], p2[0]), ('% Civilian', '% Non-USA Militant'))



plt.show()
# create columns to use when including US deaths

comp_df['Total Deaths'] = comp_df[['Civilian Deaths', 'Non-USA Militant Deaths', 'USA Militant Deaths']].sum(axis=1)

comp_df['Total Non-USA%'] = comp_df[['Civilian Deaths', 'Non-USA Militant Deaths']].sum(axis=1)/comp_df['Total Deaths']

comp_df['Total USA Militant%'] = comp_df['USA Militant Deaths']/comp_df['Total Deaths']



# plot conflicts as a bar chart

fig = plt.figure(figsize=(14, 6))

p1 = plt.bar(comp_df.index, comp_df['Total USA Militant%'],

             color='orange')

p2 = plt.bar(comp_df.index, comp_df['Total Non-USA%'],

             bottom=comp_df['Total USA Militant%'],

             color='0.75')



plt.title('US Militant Death Ratio by Conflict', fontsize=20)

plt.ylabel('% Percentage of Total Deaths')

plt.xticks(comp_df.index, comp_df['Conflict Name'])

plt.legend((p1[0], p2[0]), ('% US Militant Deaths', '% Non-US Deaths (Civilian & Militant)'))



plt.show()
#plot barchart of civilian and militant deaths over time

fig = plt.figure(figsize=(12, 6))

p1 = plt.bar(year_df['Date_clean'], year_df['Militants'], zorder=2)

p2 = plt.bar(year_df['Date_clean'], year_df['Civilians Avg'],

             bottom=year_df['Militants'], zorder=2)



plt.ylabel('People killed')

plt.xlabel('Year')

plt.xticks(year_df['Date_clean'])

plt.grid(color='0.6', linestyle='--', axis='y', zorder=1)

plt.title('USA drone kills in Pakistan by type over time', fontsize=20)

plt.legend((p1[0], p2[0]), ('Militants', 'Civilians'))



plt.show()