# Standard packages

import json



# Libs to deal with tabular data

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None

import geopandas as gpd



# Plotting packages

import seaborn as sns

import matplotlib.pyplot as plt



# Lib to create maps

import folium 

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



# To display stuff in notebook

from IPython.display import display, Markdown
# Reading Air Pollution in Seoul

stations = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')

measurements = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

items = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
# Reading GeoJSON with Seoul administrative borders

with open('/kaggle/input/maps-of-seoul/juso/2015/json/seoul_municipalities_geo_simple.json', 'r') as file:

    district_borders = json.loads(file.read())
print('Shape:', items.shape)

items
items.dtypes
# Adding unit to the item name

items['Item name (unit)'] = items['Item name'] + ' (' + items['Unit of measurement'].str.lower() + ')'



# Creating a dict of item codes to item names.

items_dict = {row['Item code']: row['Item name (unit)'] for idx, row in items.iterrows()}
# This is a function generator that creates functions to say if a measurement is good, normal, bad or very bad.

def evaluation_generator(good, normal, bad, vbad):

    def measurement_evaluator(value):

        if(pd.isnull(value) or value < 0):

            return np.nan

        elif(value <= good):

            return 'Good'

        elif(value <= normal):

            return 'Normal'

        elif(value <= bad):

            return 'Bad'

        else:

            return 'Very bad'

        

    return measurement_evaluator



# A dictionary that maps pollutants to functions that evaluate the measurement level.

evaluators = {

    row['Item name (unit)']: evaluation_generator(row['Good(Blue)'], row['Normal(Green)'], row['Bad(Yellow)'], row['Very bad(Red)']) 

    for idx, row in items.iterrows()

}
print('Shape:', stations.shape)

stations.head()
stations.dtypes
stations_dict = {row['Station code']: row['Station name(district)'] for idx, row in stations.iterrows()}
print('Shape:', measurements.shape)

measurements.sample(5, random_state=42)
measurements.dtypes
# Pivoting table to reduce number of rows

measures = measurements.pivot_table(index=['Measurement date', 'Station code', 'Instrument status'], columns='Item code', values='Average value').reset_index()

measures.columns = measures.columns.rename('')
# Replacing meaningless numbers by labels 

intrument_status = {

    0: 'Normal',

    1: 'Need for calibration',

    2: 'Abnormal',

    4: 'Power cut off',

    8: 'Under repair',

    9: 'Abnormal data',

}

measures['Instrument status'] = measures['Instrument status'].replace(intrument_status)

measures['Station code'] = measures['Station code'].replace(stations_dict)

measures = measures.rename(columns=items_dict)



# Renaming columns

measures = measures.rename(columns={

    'Measurement date': 'Date',

    'Station code': 'Station',

    'Instrument status': 'Status'

})



# Adding levels 

for pol, func in evaluators.items():

    measures[pol.split()[0] + ' Level'] = measures[pol].map(func)

    

# Casting

measures['Date'] = pd.to_datetime(measures['Date'])



# Adding date related columns

weekday_dict = {

    0:'Monday',

    1:'Tuesday',

    2:'Wednesday',

    3:'Thursday',

    4:'Friday',

    5:'Saturday',

    6:'Sunday'

}

measures['Month'] = measures['Date'].dt.month

measures['Year'] = measures['Date'].dt.year

measures['Hour'] = measures['Date'].dt.hour

measures['Day'] = measures['Date'].dt.weekday.replace(weekday_dict)
print('Shape:', measures.shape)

measures.head()
print('First date:', str(measures['Date'].min()))

print('Last date:', str(measures['Date'].max()))
stations_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)



# Add points to the map

for idx, row in stations.iterrows():

    Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(stations_map)

    

# Adding borders

folium.GeoJson(

    district_borders,

    name='geojson'

).add_to(stations_map)



# Display the map

stations_map
bad_measures = measures.loc[measures['Status'] != 'Normal', :]

all_measures = measures.copy()

measures = measures.loc[measures['Status'] == 'Normal', :]

overview = measures.groupby('Date').mean().loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']



# Adding levels 

for pol, func in evaluators.items():

    overview[pol.split()[0] + ' Level'] = overview[pol].map(func)
print('Shape:', overview.shape)

overview.sample(5, random_state=42)
fig, ax = plt.subplots(1, 6, figsize=(25, 6))

fig.suptitle('Distribution of pollutants', fontsize=16, fontweight='bold')

for n, pollutant in enumerate(evaluators.keys()):

    sns.boxplot(data = overview[pollutant], ax=ax[n])

    ax[n].set_title(pollutant)

plt.show()
general = overview.describe().loc[['min', 'max', 'mean', 'std', '25%', '50%', '75%'],:].T

general['level'] = None

for idx, row in general.iterrows():

    general.loc[idx, 'level'] = evaluators[idx](row['mean'])

    

general.T
level_counts = pd.concat([overview[col].value_counts() for col in overview.loc[:, 'SO2 Level':]], axis=1, join='outer', sort=True).fillna(0.0)

level_counts = level_counts.loc[['Very bad', 'Bad', 'Normal', 'Good'], :]



level_counts.T.plot(kind='bar', stacked=True, figsize=(8,6), rot=0,

                    colormap='coolwarm_r', legend='reverse')

plt.title('Levels of pollution in Seoul from 2017 to 2019', fontsize=16, fontweight='bold')

plt.show()
measures_slice = measures.loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']

measures_slice.columns = list(map(lambda x: x.split()[0], measures_slice.columns))

correlations = measures_slice.corr(method='spearman')

mask = np.zeros_like(correlations)

mask[np.tril_indices_from(mask)] = True



plt.figure(figsize=(8,6))

ax = sns.heatmap(data=correlations, annot=True, mask=mask, color=sns.color_palette("coolwarm", 7))

plt.title('Correlation between pollutants', fontsize=16, fontweight='bold')

plt.xticks(rotation=0) 

plt.yticks(rotation=0) 

plt.show()
district_pol = measures.groupby(['Station']).mean().loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']

district_pol_norm = (district_pol - district_pol.mean()) / district_pol.std()

district_pol_norm.columns = list(map(lambda x: x.split(' ')[0],district_pol_norm.columns))



plt.figure(figsize=(10,10))

sns.heatmap(data=district_pol_norm, cmap="YlGnBu")

plt.title('Comparision of pollutant levels across districts', fontsize=16, fontweight='bold')

plt.xticks(rotation=0) 

plt.show()
for col in district_pol_norm.columns:

    pollutant_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)



    # Add points to the map

    for idx, row in stations.iterrows():

        Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(pollutant_map)



    # Adding choropleth

    Choropleth(

        geo_data=district_borders,

        data=district_pol_norm[col], 

        key_on="feature.properties.SIG_ENG_NM", 

        fill_color='YlGnBu', 

        legend_name='Concentration of {} in Seoul (Z-score)'.format(col)

    ).add_to(pollutant_map)

    

    display(Markdown('<center><h3>{}</h3></center>'.format(col)))

    display(pollutant_map)
pollution_map = folium.Map(location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)



# Add points to the map

for idx, row in stations.iterrows():

    Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)']).add_to(pollution_map)



# Adding choropleth

Choropleth(

    geo_data=district_borders,

    data=district_pol_norm.mean(axis=1), 

    key_on="feature.properties.SIG_ENG_NM", 

    fill_color='YlGnBu', 

    legend_name='Overall pollution in Seoul by region'

).add_to(pollution_map)



pollution_map
reported_day_night = pd.Timestamp(year=2019, month=12, day=11, hour=22)



overview.loc[overview.index == reported_day_night, :]
reported_day_morning = pd.Timestamp(year=2019, month=12, day=11, hour=10)



overview.loc[overview.index == reported_day_morning, :]
measures.loc[measures['Date'] == reported_day_morning, :'PM2.5 (mircrogram/m3)'].sort_values('PM2.5 (mircrogram/m3)', ascending=False).head(10)
first_day_dec = pd.Timestamp(year=2019, month=12, day=1)

last_day_dec = pd.Timestamp(year=2019, month=12, day=31)



december = overview.loc[(overview.index >= first_day_dec) & (overview.index <= last_day_dec),:]



fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)

fig.suptitle('Pollutant concentrations on December 2019', fontsize=16, fontweight='bold')

for n, pollutant in enumerate(evaluators.keys()):

    sns.lineplot(data = december[pollutant], ax=ax[n])

    ax[n].set_title(pollutant)

plt.xlabel('Day')

plt.show()
concentration_hour = measures.groupby('Hour').mean()



fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)

fig.suptitle('Pollutant concentrations along the day', fontsize=16, fontweight='bold')

for n, pollutant in enumerate(evaluators.keys()):

    sns.lineplot(data = concentration_hour[pollutant], ax=ax[n])

    ax[n].set_title(pollutant)

plt.xlabel('Hour')

plt.show()

measures_slice = measures.loc[:, ['Date'] + list(evaluators.keys())]

measures_slice['Date'] = measures['Date'].dt.date

concentrations_day = measures_slice.groupby('Date').mean()



fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)

fig.suptitle('Pollutant concentrations along the years', fontsize=16, fontweight='bold')

for n, pollutant in enumerate(evaluators.keys()):

    sns.lineplot(data = concentrations_day[pollutant], ax=ax[n])

    ax[n].set_title(pollutant)

plt.xlabel('Date')

plt.show()
bad_measures.loc[:, 'Status':'PM2.5 (mircrogram/m3)'].sample(10, random_state=42)
# Showing 

bad_measures.loc[~bad_measures.loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)'].isnull().any(1), 'Status':'PM2.5 (mircrogram/m3)']
print('Percentage of abnormal measurements:', bad_measures.shape[0] * 100 / all_measures.shape[0])



counts = all_measures['Status'].value_counts()

plt.figure(figsize=(9,7))

plt.title('Distribution of status values', fontsize=16, fontweight='bold')

sns.barplot(x = counts.values, y = counts.index)

plt.show()
# Fails by time

#bad_measures.groupby('Year').apply(len)

#bad_measures.groupby('Month').apply(len)

bad_hourly = bad_measures.groupby('Hour').apply(len)



plt.figure(figsize=(9,7))

plt.ylabel('Quantity')

plt.xlabel('Hour')

plt.title('Quantity of abnormal measures by hour', fontsize=16, fontweight='bold')

sns.lineplot(data=bad_hourly)

plt.show()
bad_measures_hour = bad_measures.groupby(['Hour', 'Status']).apply(len).rename('Quantity').reset_index().pivot(index='Hour', columns='Status', values='Quantity').fillna(0).astype('int64')



plt.figure(figsize=(15,10))

plt.title('Distribution of the quantity of measurement fails along the day', fontsize=16, fontweight='bold')

sns.heatmap(data=bad_measures_hour, cmap="YlGnBu", annot=True, fmt='d')

plt.show()
# Fails by station

bad_stations = bad_measures['Station'].value_counts()



plt.figure(figsize=(9,7))

plt.title('Amount of times that problems occured by district', fontsize=16, fontweight='bold')

sns.barplot(x = bad_stations.values, y = bad_stations.index)

plt.show()
bad_measures_station = bad_measures.groupby(['Station', 'Status']).apply(len).rename('Quantity').reset_index().pivot(index='Station', columns='Status', values='Quantity').fillna(0).astype('int64')



plt.figure(figsize=(15,10))

sns.heatmap(data=bad_measures_station, cmap="YlGnBu", annot=True, fmt='d')

plt.title('Number of records with problems by district', fontsize=16, fontweight='bold')

plt.show()