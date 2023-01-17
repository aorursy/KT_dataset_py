import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import numpy as np

import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

print('Setup complete.')
# Function for displaying the map.

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
# Read geodata of provinces/regions.

rus_full = gpd.read_file('../input/russia-boundary-and-administrative-divisions-2015/russia_adm1_provinces_regions.shp')



rus_full.drop([65, 66], inplace=True)    # duplicates of 'Sverdlovskaya Oblast'

# Remove 'Moscow City' and 'City of St. Petersburg'.

rus_full.drop(rus_full[ rus_full['NAME_1'].str.contains('City') ].index, inplace=True)

rus_full[ rus_full['geometry'].isnull() ].tail()
from shapely import wkt



# Read pd.Series with mapping of missing regions to geometry (NL_NAME_1 -> geometry):

regions = pd.read_csv('../input/russia-crimes-2019/russia_regions_geometry.csv',

                      index_col=0, header=None, names=['geometry'])



# Convert from WKT format 'POLYGON ((42.419...' to shapely.Polygon/MultiPolygon.

regions['geometry'] = regions['geometry'].apply(wkt.loads)



# Fill missing geometry data with Polygons or MultiPolygons:

def map_regions(row):

    if row['NL_NAME_1'] in regions.index:

        row['geometry'] = regions.loc[ row['NL_NAME_1'], 'geometry' ]

    return row

# Apply function to every row.

rus_full = rus_full.apply(map_regions, axis='columns')



# So, now we have no missing geometry data.

print('Missing:', rus_full['geometry'].isnull().sum())



russia = rus_full[['NAME_1', 'NL_NAME_1', 'geometry']].set_index('NAME_1')

print('rows, cols:', russia.shape)
# Let's correct wrong data:

russia_cor_dict = {

    'Кабардино-Балкарская Респу': 'Кабардино-Балкарская Республика',

    'Карачаево-Черкессия Респуб': 'Карачаево-Черкесская Республика',

    'Респу́блика Ингуше́тия': 'Республика Ингушетия',

    'Республика Северная Осетия': 'Республика Северная Осетия',

    'Республика Чечено-Ингушска': 'Чеченская Республика',

    'Eврейская АОб': 'Еврейская автономная область',

    'Ненецкий АОк': 'Ненецкий автономный округ',

    'Ханты-Мансийский АОк': 'Ханты-Мансийский автономный округ',

    'Чукотский АОк': 'Чукотский автономный округ',

    'Ямало-Ненецкий АОк': 'Ямало-Ненецкий автономный округ',

    'Камчатская край': 'Камчатский край',

    'Пермская край': 'Пермский край',

}

russia['NL_NAME_1'].replace(russia_cor_dict, inplace=True)
russia.tail()
# Load data of criminal offences.

crimes_rus = pd.read_csv('../input/russia-crimes-2019/crimes_russia_1-10_2019.csv',

                         delimiter=';', skiprows=13, usecols=[1, 2])



# Rename columns and convert numbers from '14 220' to integer:

crimes_rus.rename(columns={crimes_rus.columns[0]: 'NL_NAME_1',

                           crimes_rus.columns[1]: 'num_crimes'}, inplace=True)

crimes_rus['num_crimes'] = crimes_rus['num_crimes'].str.replace(' ', '').astype(int)



# Keep relevant data (regions/provinces/oblast):

relevant = ['округ', 'область', 'республика', 'край']

crimes_rus = crimes_rus[ crimes_rus['NL_NAME_1'].str.contains('|'.join(relevant), case=False, regex=True) ]



# Remove federal districts (group of regions).

crimes_rus.drop(crimes_rus[ crimes_rus['NL_NAME_1'].str.contains('федеральный', case=False) ].index, inplace=True)

crimes_rus.drop(crimes_rus[ crimes_rus['NL_NAME_1']=='Республика Крым' ].index, inplace=True)

crimes_rus.head()
# Let's correct wrong data:

crimes_rus_cor_dict = {

    'Республика Саха (Якутия)': 'Республика Саха',

    'Республика Северная Осетия – Алания': 'Республика Северная Осетия',    

    'Кемеровская область – Кузбасс': 'Кемеровская область',    

    'Ханты-Мансийский автономный округ –  Югра': 'Ханты-Мансийский автономный округ',

    'Новгородская  область': 'Новгородская область',

}

crimes_rus['NL_NAME_1'].replace(crimes_rus_cor_dict, inplace=True)
# Create pd.Series for folium.Choropleth 'data' argument.

crimes_rus = crimes_rus.set_index('NL_NAME_1')['num_crimes']

crimes_rus.sort_values(ascending=False).head()
# Base map with center at city: Tyumen.

m_1 = folium.Map(location=[57.153, 65.534], tiles='cartodbpositron', zoom_start=4)



# Add a choropleth map to the base map.

Choropleth(geo_data=russia.__geo_interface__,

           data=crimes_rus,

#            key_on='feature.id',

           key_on='feature.properties.NL_NAME_1',

           fill_color='BuPu',

           legend_name='Criminal offences (Jan-Oct 2019)'

          ).add_to(m_1)



# Display the map.

embed_map(m_1, 'm_1.html')
cities = pd.read_csv("../input/world-cities-database/worldcitiespop.csv")

cities.head()
# It's not for one country, but to keep it here as simple country-to-code relation.

codes = pd.read_csv('../input/countries-iso-codes/wikipedia-iso-country-codes.csv')

codes.head()
# Consider cities with over 50 thousands of people:

russia_code = codes.loc[codes['English short name lower case']=='Russia', 'Alpha-2 code'].iloc[0].lower()

cities_rus = cities[(cities['Country']==russia_code) & (cities['Population']>50000)]

cities_rus.head()
# # Base map with center at city: Surgut.

m_2 = folium.Map(location=[61.254, 73.396], zoom_start=3)



# Add a heatmap to the base map

HeatMap(data=cities_rus[['Latitude', 'Longitude']], radius=15).add_to(m_2)



# Red color is for city with population > 1 million.

def color_producer(val):

    return 'green' if val < 1000000 else 'red'

# Add a bubble map to visualize population.

for i in range(len(cities_rus)):

    popul = cities_rus.iloc[i]['Population']

    folium.Circle(

        location=[cities_rus.iloc[i]['Latitude'], cities_rus.iloc[i]['Longitude']],

        popup='{} {}'.format(cities_rus.iloc[i]['AccentCity'], popul),

        radius=popul / 50,

        color=color_producer(popul)).add_to(m_2)



# Display the map.

embed_map(m_2, 'm_2.html')
crimes = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin')



# Drop rows with missing locations.

crimes.dropna(subset=['Lat', 'Long', 'DISTRICT'], inplace=True)



# Focus on major crimes in 2018:

crimes = crimes[crimes['OFFENSE_CODE_GROUP'].isin([

    'Aggravated Assault', 'Arson', 'Auto Theft', 'Ballistics', 'Commercial Burglary',

    'Criminal Harassment', 'HOME INVASION', 'Harassment', 'Homicide', 'Larceny',

    'Larceny From Motor Vehicle', 'Manslaughter', 'Other Burglary',

    'Residential Burglary', 'Robbery', 'Simple Assault'])]

crimes = crimes[ crimes['YEAR']==2018 ]



crimes.head()
# Firstly we'll plot how many crimes are of each kind:

crimes_amount = crimes['OFFENSE_CODE_GROUP'].value_counts()



plt.figure(figsize=(14, 6))

plt.title('Amount of crimes in Boston in 2018')

plt.xticks(rotation=45, horizontalalignment='right')

sns.set_style('whitegrid')



sns.barplot(x=crimes_amount.index, y=crimes_amount.values)
daytime_robberies = crimes[ (crimes['OFFENSE_CODE_GROUP']=='Robbery') & crimes['HOUR'].isin(range(9, 18)) ]
# Base map with center in Boston.

m_3 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)



# Add points of robberies to the map

for idx, row in daytime_robberies.iterrows():

    Marker([row['Lat'], row['Long']]).add_to(m_3)



# Display the map

embed_map(m_3, 'm_3.html')
# Base map with center in Boston.

m_4 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=13)



# Add points of robberies to the MarkerCluster() object.

mc = MarkerCluster()

for idx, row in daytime_robberies.iterrows():

    mc.add_child(Marker([row['Lat'], row['Long']]))

m_4.add_child(mc)



# Display the map

embed_map(m_4, 'm_4.html')
# Base map with center in Boston.

m_5 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)



# Add a heatmap to the base map

HeatMap(data=crimes[['Lat', 'Long']], radius=10).add_to(m_5)



# Display the map

embed_map(m_5, 'm_5.html')
# GeoDataFrame with geographical boundaries of Boston police districts.

districts_full = gpd.read_file('../input/boston-police-districts/Police_Districts.shp')

districts = districts_full[['DISTRICT', 'geometry']].set_index('DISTRICT')

districts.head()
# We also create pd.Series that shows the number of crimes in each district with the same index. 

# This is how the code knows how to match the geographical boundaries with appropriate colors.

plot_dict = crimes['DISTRICT'].value_counts()

plot_dict.head()
# Base map with center in Boston.

m_6 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=12)



# Add a choropleth map to the base map.

Choropleth(geo_data=districts.__geo_interface__,

           data=plot_dict,

           key_on='feature.id',

           fill_color='BuPu',

           legend_name='Major criminal incidents (Jan-Aug 2018)'

          ).add_to(m_6)



# Display the map

embed_map(m_6, 'm_6.html')