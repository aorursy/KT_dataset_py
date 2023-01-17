import os
running_in_kaggle = os.getenv('KAGGLE_WORKING_DIR') is not None
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import json
import requests

import folium
import seaborn as sns

%matplotlib inline
import matplotlib.pyplot as plt
# Data manipulation
# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().magic(u'load_ext autoreload')

get_ipython().magic(u'autoreload 2')
# https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e
# Base Path
base_path = Path.cwd()

# Data paths
data_path = Path('../input') if running_in_kaggle else base_path / 'data'
raw_data_path =  data_path / 'walmart-supply-chain-data'  if running_in_kaggle else data_path / 'raw'
interim_data_path = data_path / 'interim'
processed_data_path = data_path / 'processed'
external_data_path = data_path / 'external'

supply_chain_data_raw_fname = 'walmart-import-data-full.csv'
supply_chain_data_raw_path = raw_data_path / supply_chain_data_raw_fname

kaggle_data_home = 'https://www.kaggle.com/sunilp/walmart-supply-chain-data/downloads/supply-chain-data.zip/1'

# Reports paths
reports_path = base_path / 'reports'
figures_path = reports_path / 'figures'

# Input paths

# Folium Geojson Data
folium_world_countries_fname = 'world-countries.json'
folium_example_data_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
folium_world_countries_url = f'{folium_example_data_url}/{folium_world_countries_fname}'
# Local paths for Folium data
folium_world_countries_path = data_path / 'world-countries' / folium_world_countries_fname

folium_world_data = external_data_path / folium_world_countries_fname
# Constants and Globals

# Source: http://www.dsv.com/sea-freight/sea-container-description/dry-container
TW_KG_DRY_CONTAINER_20FT = 2300 # Tare weight, i.e. weight of the container itself
TW_KG_DRY_CONTAINER_40FT = 3750

NW_KG_DRY_CONTAINER_20FT = 25000 # Net weight, i.e. weight of the payload in the container
NW_KG_DRY_CONTAINER_40FT = 27600

GW_KG_DRY_CONTAINER_20FT = (TW_KG_DRY_CONTAINER_20FT + NW_KG_DRY_CONTAINER_20FT)
GW_KG_DRY_CONTAINER_40FT = (TW_KG_DRY_CONTAINER_40FT + NW_KG_DRY_CONTAINER_40FT)

CAPACITY_CONTAINER_40FT = 67.7 # cubic capacity, in m^3

US_STD_GALLONS_PER_BBL = 42
LITERS_PER_US_STD_GALLON = 3.785411784 # https://en.wikipedia.org/wiki/Gallon
LITERS_PER_BBL = (US_STD_GALLONS_PER_BBL * LITERS_PER_US_STD_GALLON)

# http://www.etc-cte.ec.gc.ca/databases/oilproperties/pdf/WEB_Gasoline_Blending_Stocks_(Alkylates).pdf
ALKYLATE_DENSITY_15C = 0.7090 # g/mL or Kg/L
ALKYLATE_DENSITY_38C = 0.6890 # g/mL or Kg/L
def coo_freq(xdata):
    total = sum(xdata['COUNTRY OF ORIGIN'].value_counts())
    coo_dict = xdata['COUNTRY OF ORIGIN'].value_counts().to_dict()
    return xdata['COUNTRY OF ORIGIN'].apply(lambda cn: 100*coo_dict.get(cn,0)/total)

# See https://github.com/python-visualization/folium/blob/master/examples/GeoJSON_and_choropleth.ipynb
# We don't need the GeoJSON data locally, we can just provide a URL
def get_folium_map_data():
    if running_in_kaggle:
        with open(folium_world_countries_path) as fh:
            folium_world_data = json.load(fh)
    else:
        folium_world_data = json.loads(requests.get(folium_world_countries_url).text)
    return folium_world_data

def calc_kg_per_container(row):
    # If container count is 0, use quantity instead
    # alternative would be to ignore these
    cc = row['CONTAINER COUNT']
    if cc == 0:
        cc = row['QUANTITY']
    return row['WEIGHT (KG)']/max(1, row['CONTAINER COUNT'])
folium_world_data = get_folium_map_data()

if not running_in_kaggle and not supply_chain_data_raw_path.exists():
    # Make sure the raw Kaggle data exists locally
    print('Download the data from Kaggle here: {}'.format(kaggle_data_home))
    print('After downloading, move the file {}\nhere: {}'.format(supply_chain_data_raw_fname, raw_data_path))

if supply_chain_data_raw_path.exists():
    xdata = pd.read_csv(supply_chain_data_raw_path, low_memory=False)
    # About 9% of the rows are all NA; drop them
    xdata.dropna(axis=0, how='all', inplace=True)
else:
    print('Failed to load data')
    assert False
xdata.head()
xdata.shape
xdata.columns
xdata['COUNTRY OF ORIGIN'].describe().to_dict()
countries_of_origin = xdata.groupby('COUNTRY OF ORIGIN').describe().xs('WEIGHT (KG)', axis=1).copy()
countries_of_origin.reset_index(level=0, inplace=True)
coo_record_count = sum(countries_of_origin['count'])
countries_of_origin['pct'] = countries_of_origin['count'].apply(lambda cnt: 100*cnt/coo_record_count)
countries_of_origin['logpct'] = countries_of_origin['count'].apply(lambda cnt: np.log(100*cnt/coo_record_count))
# We will use the sorting later
countries_of_origin.sort_values(by=['count', 'max'], ascending=False)
# Initialize the map:
m = folium.Map(location=[30, 0], zoom_start=2, no_wrap=True)

# geo_data is the map data

# Add the color for the chloropleth:
_ = folium.Choropleth(
    geo_data=folium_world_data,
    name='Choropleth',
    data=countries_of_origin,
    columns=['COUNTRY OF ORIGIN', 'pct'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color='white',
    legend_name='Country of Origin (%)'
).add_to(m)
 
if not running_in_kaggle:
    # Save to html
    map_path = reports_path / 'walmart_folium_chloropleth_pct.html'
    m.save(map_path.as_posix())
    
m
# Initialize the map:
m = folium.Map(location=[30, 0], zoom_start=2, no_wrap=True)

# geo_data is the map data

# Add the color for the chloropleth:
_ = folium.Choropleth(
    geo_data=folium_world_data,
    name='Choropleth',
    data=countries_of_origin,
    columns=['COUNTRY OF ORIGIN', 'logpct'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color='white',
    legend_name='Country of Origin (log %)'
).add_to(m)
 
if not running_in_kaggle:
    # Save to html
    map_path = reports_path / 'walmart_folium_chloropleth_logpct.html'
    m.save(map_path.as_posix())
    
m
xdata[xdata['COUNTRY OF ORIGIN']=='Saudi Arabia']['PRODUCT DETAILS'].describe().to_dict()
countries_of_origin.sort_values(by=['count'], axis=0, ascending=False).tail(4)
top_by_weight = countries_of_origin.sort_values(by=['max'], axis=0, 
                                                ascending=False)[0:10]['COUNTRY OF ORIGIN'].values

# See: https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook-grouping

tw3 = xdata.groupby(['COUNTRY OF ORIGIN']).apply(
    lambda subf: 
        subf.sort_values('WEIGHT (KG)', 
                         ascending=False).head(3))

cols_of_interest = ['COUNTRY OF ORIGIN', 'WEIGHT (KG)', 'PRODUCT DETAILS', 'ARRIVAL DATE']
# https://www.wolframalpha.com/input/?i=3.75e%2B05+kg
#  3.75e+05 kg ≈ 1.001 × maximum takeoff mass of a Boeing 747 aircraft ( 412.8 sh tn )
max_takeoff_wt_747 = 3.75e+05 #kg

# https://www.wolframalpha.com/input/?i=3e%2B07+kg
#  ≈ (0.7 to 1) × mass of a Handy size cargo ship ( 28000 to 40000 lg tn )

mass_handy_size = 3.0e+07 #kg
tw3.loc[(tw3['COUNTRY OF ORIGIN'].isin(top_by_weight)) & (tw3['WEIGHT (KG)']>mass_handy_size)][cols_of_interest]
# Because the dataframe display truncates the 'PRODUCT DETAILS' column, lets make sure we are not talking about olive oil here...
list(xdata[xdata['WEIGHT (KG)']>1e+09]['PRODUCT DETAILS'])[0]
# Petroleum naphtha is an intermediate hydrocarbon liquid stream derived from the refining of crude oil
# NB 4.340600e+07 => 43406 t (metric tons)
#  ≈ (1.1 to 1.5) × mass of a Handy size cargo ship ( 28000 to 40000 lg tn )
# https://www.wolframalpha.com/input/?i=4.340600e%2B07+kg
# Kuwait	187193	Kuwait	4.340600e+07	GRANULAR UREA IN BULK

# https://en.wikipedia.org/wiki/Handysize
# Handysize is a naval architecture term for smaller bulk carriers or oil tanker with deadweight of up to 50,000 tonnes

# Heaviest item
# https://www.wolframalpha.com/input/?i=5.550023e%2B09+kg
# 5.550023e+09 kg ≈ 0.93 × mass of the Great Pyramid of Giza (≈ 6×10^9 kg )
xdata['log_weight_kg'] = xdata['WEIGHT (KG)'].apply(lambda x: np.log(max(1, x)))
xdata['WEIGHT (KG)'].describe()
num_bins = 25 
n, bins, patches = plt.hist(list(xdata['log_weight_kg']), num_bins, facecolor='blue', alpha=0.5)
_ = plt.xlabel('log(weight (Kg)) of shipment');
_ = plt.ylabel('Count');
_ = plt.title('Distribution of Weights of Shipments');
_ = plt.axvline(x=xdata['log_weight_kg'].median(), color='r', linestyle='--')

if running_in_kaggle:
    plt.show();
else:
    plt_path = figures_path / 'log_weight_kg.png'
    plt.savefig(plt_path.as_posix(), format='png')
xdata['kg_per_container'] = xdata.apply(calc_kg_per_container, axis = 1)
containerizable = xdata[xdata['kg_per_container'] <= NW_KG_DRY_CONTAINER_40FT]
containerizable.shape
containerizable_pct = 100*containerizable.shape[0]/xdata.shape[0]
print('Percent of shipments meet weight requirements for a 40 ft container: {:0.2f} %'.format(containerizable_pct))
num_bins = 100
xs = containerizable['kg_per_container']
n, bins, patches = plt.hist(xs, num_bins, facecolor='blue', alpha=0.5, density=False)
_ = plt.xlabel('Kg Per Container');
_ = plt.ylabel('Count');
_ = plt.title('Containerizable Shipments');

if running_in_kaggle:
    plt.show();
else:
    plt_path = figures_path / 'containerizable.png'
    plt.savefig(plt_path.as_posix(), format='png')
xdata['M.UNIT'].value_counts().to_dict()
low_units = ['F', 'CC', 'MM', 'SS', 'XX', 'FF']
cols_of_interest = ['COUNTRY OF ORIGIN', 'WEIGHT (KG)', 'QUANTITY', 'CONTAINER COUNT',
                    'MEASUREMENT', 'M.UNIT', 'PRODUCT DETAILS'] # , 'ARRIVAL DATE'
pd.set_option('display.max_colwidth', 125)
pd.set_option('display.max_rows', 50)
xdata[xdata['M.UNIT'].isin(low_units)][cols_of_interest]
xic = xdata[(xdata['CONTAINER COUNT']>0) & (xdata['WEIGHT (KG)']>0) & 
            (xdata['kg_per_container']<=NW_KG_DRY_CONTAINER_40FT)]
# N.B. A similar plot with swarmplot ran for over an hour without completing, see
# https://github.com/mwaskom/seaborn/issues/1176
g = sns.stripplot(x='MEASUREMENT', y='M.UNIT', data=xic, jitter=True)
sns.despine() # remove the top and right line in graph
g.figure.set_size_inches(8,6)

if running_in_kaggle:
    plt.show();
else:
    plt_path = figures_path / 'measure_stripplot.png'
    plt.savefig(plt_path.as_posix(), format='png')
xdata_cm = xdata[(xdata['M.UNIT'] == 'CM') & (xdata['MEASUREMENT'] > 0.0)].copy()
xdata_cm['logmeasure'] = xdata_cm['MEASUREMENT'].apply(lambda mm: np.log10(mm))
xdata_cm['MEASUREMENT'].describe()
max_logmeasure = xdata_cm.logmeasure.max()
max_logmeasure_shipment_id = xdata_cm['logmeasure'].idxmax()

g = sns.boxplot(x = xdata_cm['logmeasure'])
# remove the top and right line in graph
sns.despine()

_ = plt.annotate(s = max_logmeasure_shipment_id,
             xy = (max_logmeasure,0),
             xytext=(0.85, 0.65), textcoords='axes fraction', # bottom, left
             # Shrink the arrow to avoid occlusion
             arrowprops = {'facecolor':'gray', 'width': 1, 'shrink': 0.09, 'headlength':9},
             backgroundcolor = 'white')

# g.figure.set_size_inches(6,4)
g.figure.set_size_inches(8,6)
if running_in_kaggle:
    plt.show();
else:
    plt_path = figures_path / 'logmeasure_box.png'
    plt.savefig(plt_path.as_posix(), format='png')
num_bins = 25 
n, bins, patches = plt.hist(list(xdata_cm['logmeasure']), num_bins, facecolor='blue', alpha=0.5)
_ = plt.xlabel('log10(measurement)');
_ = plt.ylabel('Count');
_ = plt.title('Distribution of log10(measurement)');
_ = plt.axvline(x=xdata_cm['logmeasure'].median(), color='r', linestyle='--')

if running_in_kaggle:
    plt.show();
else:
    plt_path = figures_path / 'logmeasure.png'
    plt.savefig(plt_path.as_posix(), format='png')
cols_of_interest = ['SHIPPER', 'WEIGHT (KG)', 'QUANTITY', 'MEASUREMENT', 'CONTAINER COUNT', 'PRODUCT DETAILS']
xdata_cm.sort_values(by='logmeasure', ascending=False)[cols_of_interest].head(10)
xdata_cm_sm = xdata_cm[xdata_cm['MEASUREMENT'] < CAPACITY_CONTAINER_40FT]
xdata_cm_sm.sort_values(by='logmeasure', ascending=False)[cols_of_interest].head(10)
xdata['WEIGHT (KG)'].max()
xdata['ARRIVAL DATE'].describe()
alkylate = xdata[xdata['WEIGHT (KG)']>1e+09] #['PRODUCT DETAILS']
alkylate
# 42 U.S. standard gallons to liters => 4672 L (liters)
bahamian_entries = xdata[xdata['COUNTRY OF ORIGIN']=='Bahamas']
_ = bahamian_entries.sort_values('WEIGHT (LB)', ascending=False)['PRODUCT DETAILS']
bahamian_entries.shape
bahamian_entries.columns
bahamian_entries[bahamian_entries['WEIGHT (KG)'] > 2*max_takeoff_wt_747]
bah_187736 = bahamian_entries.loc[187736].to_dict()
bah_187736['PRODUCT DETAILS']
import re
rex = re.compile('([0-9]*\.[0-9]*)')
bbls = float(rex.search(bah_187736['PRODUCT DETAILS'])[0])
bbls
US_STD_GALLONS_PER_BBL = 42
LITERS_PER_US_STD_GALLON = 3.785411784 # 
LITERS_PER_BBL = (US_STD_GALLONS_PER_BBL * LITERS_PER_US_STD_GALLON)
bah_187736_liters = (bbls * LITERS_PER_BBL)

# http://www.etc-cte.ec.gc.ca/databases/oilproperties/pdf/WEB_Gasoline_Blending_Stocks_(Alkylates).pdf
ALKYLATE_DENSITY_15C = 0.7090 # g/mL or Kg/L
ALKYLATE_DENSITY_38C = 0.6890 # g/mL or Kg/L

print(bah_187736_liters, bah_187736_liters * pd.Series([ALKYLATE_DENSITY_15C, ALKYLATE_DENSITY_38C]))
bah_187736['WEIGHT (KG)']
# Assume linearity in range between 15C and 38C
# x axis is temperature (C), y axis is density (Kg/L)
slope = (ALKYLATE_DENSITY_38C - ALKYLATE_DENSITY_15C)/(38-15)
intercept = -38*slope + ALKYLATE_DENSITY_38C
xs = range(14, 39, 1)
ys = slope*pd.Series(xs) + intercept
plt.plot(xs, ys);
weight = bah_187736['WEIGHT (KG)']
density = weight / bah_187736_liters
bah_187736_temp = (density - intercept)/slope
bah_187736_temp
# mean temperature bahamas
# bahamas historical temperatures

# https://www.timeanddate.com/weather/bahamas/nassau/historic?month=1&year=2013
# ship transit time bahamas to new york
# http://ports.com/sea-route/port-of-new-york,united-states/port-of-nassau,bahamas/

# We have the date the shipment arrived in NY
bah_187736['FOREIGN PORT'], bah_187736['US PORT']
# http://www.worldportsource.com/ports/BHS_Port_of_South_Riding_Point_1322.php
# Freeport, Grand Bahama, Bahamas

# http://ports.com/sea-route/port-of-new-york,united-states/port-of-nassau,bahamas/#/?a=0&b=1855&c=Port%20of%20New%20York,%20United%20States&d=Port%20of%20South%20Riding%20Point,%20Bahamas
# http://ports.com/sea-route/
# Port of South Riding Point

# give time at sea of 4.4 days at 10 knots
# 3.2 days at 14 knots
# https://en.wikipedia.org/wiki/Bulk_carrier

bah_187736['ARRIVAL DATE']
# looking back 4 days from the arrival date, we see the temperature in Nassau, Bahamas 
# has a low of 21C and a high of 27C, so this seems plausible
# Since density declines with increasing temperature, a measurement at a higher 
# temperature would show a lower weight, so it is unlikely that the shipment is short.
weight/ALKYLATE_DENSITY_15C
bah_187736_liters
(weight/ALKYLATE_DENSITY_15C)/bah_187736_liters
bah_187736_liters/(weight/ALKYLATE_DENSITY_15C)