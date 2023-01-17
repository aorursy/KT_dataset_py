import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import bq_helper
from mpl_toolkits.basemap import Basemap
%matplotlib inline
open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data', 
                                  dataset_name='openaq')
open_aq.list_tables()
open_aq.table_schema('global_air_quality')
open_aq.head('global_air_quality')
query = """
        SELECT unit, COUNT(pollutant) as value
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY unit
"""

open_aq.estimate_query_size(query)
unit_distribution = open_aq.query_to_pandas_safe(query)
print(unit_distribution)

sns.barplot(x='unit',y='value',data=unit_distribution)
plt.title('Unit distribution')
query = """
        SELECT COUNT(unit) as ppm_observation, pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'ppm'
        GROUP BY pollutant
        
"""
open_aq.estimate_query_size(query)
pollutant_unit_ppm = open_aq.query_to_pandas_safe(query)
print(pollutant_unit_ppm)
sns.barplot(x='pollutant',y='ppm_observation', data=pollutant_unit_ppm)
plt.title('PPM unit distribution')
query = """
        SELECT COUNT(unit) as mcubic_observation, pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        GROUP BY pollutant
"""
open_aq.estimate_query_size(query)
pollutant_unit_mg3 = open_aq.query_to_pandas_safe(query)
print(pollutant_unit_mg3)
sns.barplot(x='pollutant',y='mcubic_observation',data=pollutant_unit_mg3)
plt.title('µg/m³ unit distribution')
query = """
        SELECT COUNT(unit) as number_of_observations, pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
"""

open_aq.estimate_query_size(query)
number_of_observ = open_aq.query_to_pandas_safe(query)
sns.barplot(x='pollutant',y='number_of_observations',data=number_of_observ)
plt.xlabel('pollutant')
plt.ylabel('number of observations')
plt.title('Total number of observations')
query = """
        SELECT AVG(value) as Average, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'pm25' and value > 0
        GROUP BY country
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
pm25_country_pollution = open_aq.query_to_pandas_safe(query)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=pm25_country_pollution['Average'],x=pm25_country_pollution['country'])
plt.title('Average PM 2.5 air pollution by country(µg/m³)')
query = """
    SELECT value, unit, pollutant, country
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE country = 'SG' and pollutant = 'pm25' and value > 0
"""
open_aq.estimate_query_size(query)
sg_pm25_dist = open_aq.query_to_pandas_safe(query)
sg_pm25_dist
query = """
        SELECT AVG(value) as Average, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'pm25' and value > 0
        GROUP BY city, latitude, longitude
"""
open_aq.estimate_query_size(query)
pm25_map = open_aq.query_to_pandas_safe(query)
pm25_map['Average'].plot()
outliers = pm25_map[pm25_map['Average'] > 1500]
outliers
pm25_map = pm25_map.drop(pm25_map.index[pm25_map['Average'] > 1500])
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(pm25_map['longitude'],pm25_map['latitude'],c=pm25_map['Average'],alpha=0.7,cmap='Reds')
plt.title('Average PM 2.5 Air Pollution by location(µg/m³)')
plt.colorbar()
query = """
        SELECT AVG(value) as Average, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'pm10' and value > 0
        GROUP BY country
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
pm10_country_pollution = open_aq.query_to_pandas_safe(query)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=pm10_country_pollution['Average'],x=pm10_country_pollution['country'])
plt.title('Average PM 10 air pollution by country(µg/m³)')
query = """
        SELECT AVG(value) as Average, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'pm10' and value > 0
        GROUP BY city, latitude, longitude
"""
open_aq.estimate_query_size(query)
pm10_map = open_aq.query_to_pandas_safe(query)
pm10_map['Average'].plot()
outliers = pm10_map[pm10_map['Average'] > 1500]
outliers
pm10_map = pm10_map.drop(pm10_map.index[pm10_map['Average'] > 1500])
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(pm10_map['longitude'],pm10_map['latitude'],c=pm10_map['Average'],alpha=0.7,cmap='Oranges')
plt.title('Average PM 2.5 Air Pollution by location(µg/m³)')
plt.colorbar()
#molecular weight for pollutants
molar_volume_const = 24.45
NO2_mol_weight = 46.00550
o3_mol_weight = 48
so2_mol_weight = 64.066
co_mol_weight = 28.01
# conversion check
# http://www.aresok.org/npg/nioshdbs/calc.htm  
# https://www.metric-conversions.org/weight/milligrams-to-micrograms.htm
#function for converting ppm to µg/m³
def gas_convertion(ppm_unit, mol_mass):
    return (ppm_unit * mol_mass / molar_volume_const) / 0.001
query = """
        SELECT AVG(value) as Average, unit, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'no2' and value > 0
        GROUP BY country, unit
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
no2_country_pollution = open_aq.query_to_pandas_safe(query)
no2_country_pollution
#same counties that have diffrent units
print(no2_country_pollution[no2_country_pollution['country'].duplicated(keep=False)])
#converting units
no2_country_pollution['Average'] = no2_country_pollution.apply(lambda x: gas_convertion(x['Average'], NO2_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
print(no2_country_pollution)
no2_country_pollution = no2_country_pollution.groupby('country', as_index=False).mean().sort_values(by='Average',axis=0)

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=no2_country_pollution['Average'],x=no2_country_pollution['country'])
plt.title('Average NO2 air pollution by country(µg/m³)')
query = """
        SELECT AVG(value) as Average, unit, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'no2' and value > 0
        GROUP BY city, unit, latitude, longitude
"""
open_aq.estimate_query_size(query)
no2_map = open_aq.query_to_pandas_safe(query)
no2_map['Average'].plot()
no2_map['Average'] = no2_map.apply(lambda x: gas_convertion(x['Average'], NO2_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(no2_map['longitude'],no2_map['latitude'],c=no2_map['Average'],alpha=0.7,cmap='Reds')
plt.title('Average NO2 Air Pollution by location(µg/m³)')
plt.colorbar()
query = """
        SELECT AVG(value) as Average, unit, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'o3' and value > 0
        GROUP BY country, unit
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
o3_country_pollution = open_aq.query_to_pandas_safe(query)
o3_country_pollution
print(o3_country_pollution[o3_country_pollution['country'].duplicated(keep=False)])
o3_country_pollution['Average'] = o3_country_pollution.apply(lambda x: gas_convertion(x['Average'], o3_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
o3_country_pollution = o3_country_pollution.groupby('country', as_index=False).mean().sort_values(by='Average',axis=0)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=o3_country_pollution['Average'],x=o3_country_pollution['country'])
plt.title('Average O3 air pollution by country(µg/m³)')
query = """
        SELECT AVG(value) as Average, unit, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'o3' and value >= 0
        GROUP BY city, unit, latitude, longitude
"""
open_aq.estimate_query_size(query)
o3_map = open_aq.query_to_pandas_safe(query)
o3_map['Average'] = o3_map.apply(lambda x: gas_convertion(x['Average'], o3_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
o3_map['Average'].plot()
outliers = o3_map[o3_map['Average'] > 300]
outliers
o3_map = o3_map.drop(o3_map.index[o3_map['Average'] > 300])
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(o3_map['longitude'],o3_map['latitude'],c=o3_map['Average'],alpha=0.7,cmap='Reds', vmax=250)
plt.title('Average O3 Air Pollution by location(µg/m³)')
plt.colorbar()
query = """
        SELECT AVG(value) as Average, unit, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'so2' and value > 0
        GROUP BY country, unit
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
so2_country_pollution = open_aq.query_to_pandas_safe(query)
so2_country_pollution
print(so2_country_pollution[so2_country_pollution['country'].duplicated(keep=False)])
so2_country_pollution['Average'] = so2_country_pollution.apply(lambda x: gas_convertion(x['Average'], so2_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
so2_country_pollution = so2_country_pollution.groupby('country', as_index=False).mean().sort_values(by='Average',axis=0)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=so2_country_pollution['Average'],x=so2_country_pollution['country'])
plt.title('Average SO2 air pollution by country(µg/m³)')
query = """
        SELECT value, unit, pollutant, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'AU' and pollutant = 'so2' and value > 0
"""
open_aq.estimate_query_size(query)
au_pollution = open_aq.query_to_pandas_safe(query)
au_pollution
au_pollution.plot()
query = """
        SELECT value, unit, pollutant, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'CL' and pollutant = 'so2' and value > 0
"""
open_aq.estimate_query_size(query)
cl_pollution = open_aq.query_to_pandas(query)
cl_pollution
cl_pollution.plot()
query = """
        SELECT AVG(value) as Average, unit, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'so2' and value >= 0
        GROUP BY city, unit, latitude, longitude
"""
open_aq.estimate_query_size(query)
so2_map = open_aq.query_to_pandas_safe(query)
so2_map['Average'] = so2_map.apply(lambda x: gas_convertion(x['Average'], so2_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)

outliers = so2_map[so2_map['Average'] > 1500]
outliers
so2_map = so2_map.drop(so2_map.index[so2_map['Average'] > 700])
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(so2_map['longitude'],so2_map['latitude'],c=so2_map['Average'],alpha=0.7,cmap='Reds')
plt.title('Average so2 Air Pollution by location(µg/m³)')
plt.colorbar()
query = """
        SELECT AVG(value) as Average, unit, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'co' and value > 0
        GROUP BY country, unit
        ORDER BY Average
"""
open_aq.estimate_query_size(query)
co_country_pollution = open_aq.query_to_pandas_safe(query)
co_country_pollution
co_country_pollution['Average'] = co_country_pollution.apply(lambda x: gas_convertion(x['Average'], co_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)


co_country_pollution = co_country_pollution.groupby('country', as_index=False).mean().sort_values(by='Average',axis=0)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.barplot(y=co_country_pollution['Average'],x=co_country_pollution['country'])
plt.title('Average CO air pollution by country(µg/m³)')
query = """
        SELECT value, unit, pollutant, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'TH' and pollutant = 'co'
"""
open_aq.estimate_query_size(query)
thai_pollution = open_aq.query_to_pandas_safe(query)
thai_pollution
thai_pollution.plot()
query = """
        SELECT value, unit, pollutant, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'PT' and pollutant = 'co'
"""
open_aq.estimate_query_size(query)
portugal_pollution = open_aq.query_to_pandas_safe(query)
portugal_pollution
portugal_pollution.plot()
query = """
        SELECT AVG(value) as Average, unit, city, latitude, longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = 'co' and value >= 0
        GROUP BY city, unit, latitude, longitude
"""
open_aq.estimate_query_size(query)
co_map = open_aq.query_to_pandas_safe(query)
co_map['Average'] = co_map.apply(lambda x: gas_convertion(x['Average'], co_mol_weight)
                            if x['unit'] == 'ppm' else x['Average'], axis=1)
outliers = co_map[co_map['Average'] > 8000]
outliers
co_map = co_map.drop(co_map.index[co_map['Average'] > 8000])
fig = plt.figure(figsize=(16,10))
bmap = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90,urcrnrlat=90,
                resolution='c', lat_ts=True)
bmap.shadedrelief()
bmap.drawmapboundary(fill_color='#9ad3de', linewidth=0.1)
bmap.fillcontinents(color='#daad86', alpha=0.2)
bmap.drawcoastlines(linewidth=0.1, color='#312c32')

bmap.scatter(co_map['longitude'],co_map['latitude'],c=co_map['Average'],alpha=0.7,cmap='Reds')
plt.title('Average CO Air Pollution by location(µg/m³)')
plt.colorbar()