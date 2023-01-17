# Set up environment
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd

# For Windows: To address problem with current windows build of GeoPandas
# https://github.com/geopandas/geopandas/issues/830
#import os
#os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share" #windows
# Find file locations on Kaggle
import os
os.listdir("../input/")
#Read in law enforcement agency (LEA) shapefile and check projection
#lea_shape = gpd.read_file("../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Shapefiles/SFPD_geo_export_8a7e3c3b-8d43-4359-bebe-b7ef6a6e21f3.shp")
lea_shape = gpd.read_file("../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Shapefiles/SFPD_geo_export_8a7e3c3b-8d43-4359-bebe-b7ef6a6e21f3.shp")
print(lea_shape.crs)
print(lea_shape.shape)
lea_shape.head()
# Read in shape file for census tracts for the state. 
# This file can also be accessed directly from the Census Bureau via URL. 
# Tract maps for counties are also available, but some police departments might span counties (e.g., NYC).
state_shape = gpd.read_file("../input/ca-tract-shapefile/cb_2017_06_tract_500k.shp")
#state_shape = gpd.read_file("./cb_2017_06_tract_500k/cb_2017_06_tract_500k.shp")

print(state_shape.crs)
print(state_shape.shape)
state_shape.head()
# Convert LEA geometry to same projection as Census files
lea_shape = lea_shape.to_crs({'init': 'epsg:4269'}) 
# Quick plot shows the 10 police districts
lea_shape.plot(column = 'district')
# Plot both LEA districts and census tracts together

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Overlay of San Francisco Police Districts with Census Tracts")

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
plt.show();
# Create new Geodataframe that contains the intersections
# Will not work on Kaggle
#intersect = gpd.overlay(state_shape, lea_shape, how='intersection')
# for Kaggle
# Export intersect dataframe so it can be used on Kaggle platform
#intersect.to_pickle("./Dept_49-00081/Dept_49-00081_intersect.pkl")

# Read in pickled dataframe with the results of the overlay operation
intersect = pd.read_pickle("../input/intersection/Dept_49-00081_intersect.pkl")

intersect.TRACTCE.value_counts().head()
# Select tract that has been split into 4 parts
tract_new = intersect[intersect['TRACTCE'] == "016801"]

# Show shape of original tract
tract_old = state_shape[state_shape['TRACTCE'] == "016801"]

figure1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
figure1.set_size_inches(10, 4, forward=True)
ax1.set_title('Original census tract 016801')
ax1.set_yticks([0.001, 0.003])

# Show tract split by intersection
tract_old.plot(ax=ax1, color='white', edgecolor='black')
tract_new.plot(ax=ax2, color='white', edgecolor='black')
ax2.set_title("Intersection of census tract with LEA districts")
plt.show()

# Get area of intersection polygons 
intersect['inter_area'] = intersect.area
# And drop unneeded columns
#intersect = intersect.drop(columns = ['ALAND', 'AWATER', 'shape_area', 'shape_le_1', 'shape_leng'])
# Get area of each tract by summing its polygons
tract_area = intersect.groupby("TRACTCE").agg({'inter_area': 'sum'}).rename(columns={'inter_area': 'tract_area'})
tract_area.head(2)
# Get area of each LEA by summing its polygons
LEA_area = intersect.groupby("company").agg({'inter_area': 'sum'}).rename(columns={'inter_area': 'LEA_area'})

# Merge intersection dataframe with tract and LEA area data
intersect_area = intersect.merge(tract_area, how='left', on='TRACTCE').merge(LEA_area, how='left', on='company')

# Calculate polygon percent of tract and polygon percent of LEA district
intersect_area['prop_of_tract'] = intersect_area.inter_area / intersect_area.tract_area
intersect_area['prop_of_LEA'] = intersect_area.inter_area / intersect_area.LEA_area
intersect_area.info()
# Read in ACS 2015 5yr DP05 file from Kaggle
file = '../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_ACS_data/49-00081_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv'
dp05 = pd.read_csv(file, skiprows=[1])[['HC01_VC03', 'GEO.id', 'GEO.id2', 'GEO.display-label']].rename(columns={'HC01_VC03': 'totpop'})

# Extract tract number from text string
split1 = dp05['GEO.display-label'].str.split(',', expand=True)
split2 = split1[0].str.split(' ', expand=True)
dp05['tract_num'] = split2[2]

dp05.head()
# Merge on NAME (instead of TRACTCE) because it is an easier match to the census data
intersect_data = intersect_area.merge(dp05, how='left', left_on='NAME', right_on='tract_num')

intersect_data[intersect_data['tract_num'].isna()]
# Get rid of LEA overlap polygons outside of San Francisco County
intersect_data = intersect_data[intersect_data['tract_num'].notna()]

# Estimate population in each intersection polygon
# Assumes that each polygon has the same proportion of population as it has of land area.
intersect_data['poly_pop'] = intersect_data.totpop * intersect_data.prop_of_tract

# Aggregate intersection polygons to the LEA district level
LEA_pop = intersect_data.groupby("company").agg({'poly_pop': 'sum'}).rename(columns={'poly_pop': 'totpop'})
#LEA_pop.info()
# Check to see if LEA data has same total as original census data.
print("San Francisco population from original census data: ", dp05['totpop'].sum() )
print("San Francisco population from new LEA data: ", LEA_pop['totpop'].sum() )
# Merge LEA-level census data to LEA geo data for mapping
lea_shape_data = lea_shape.merge(LEA_pop, how='left', on='company')
# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Population estimates for San Francisco Police Districts, ACS 5yr 2015", fontsize=15)

lea_shape_data.plot(ax=ax, column='totpop', cmap='BuPu', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(lea_shape_data['totpop']),  np.max(lea_shape_data['totpop'])
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();
# One way to access the Census Bureau API is through the Python package CensusData
# https://github.com/jtleider/censusdata
#import censusdata
#import pandas as pd
#from sodapy import Socrata

# The search method can be used to identify tables containing keyword of interest
# Just an example of how to search for tables. This will produce long output.
#censusdata.search('acs1', '2015', 'label', 'housing')

# The printtable method can get field names and descriptions from a specific table. 
# Also produces long output. Running the code below will retreive documentation of the DP05 table
# censusdata.printtable(censusdata.censustable('acs5', '2015', 'DP05'))

# Method to get list of available geographies
#censusdata.geographies(censusdata.censusgeo([('state', '06'), ('county', '075'), ('tract', '*')]), 'acs5', 2015)
# Will not work on Kaggle
# Define function to retrieve census data
def get_census(file, year, geography, census_vars):
    census_raw = censusdata.download(file, year, geography, census_vars, tabletype='profile') 
    census_raw.reset_index(inplace = True)
    census_raw['tract_string'] = census_raw['index'].astype(str)
    # Extract tract number from text string
    split1 = census_raw['tract_string'].str.split(',', expand=True)
    split2 = split1[0].str.split(' ', expand=True)
    census_raw['tract_num'] = split2[2]
    census_raw = census_raw.drop(columns=['index'])
    return census_raw

# Will not work on Kaggle
# Define parameters for census download
year = 2015
file = 'acs5'
geography = censusdata.censusgeo([('state', '06'), ('county', '075'), ('tract', '*')])
census_vars = ['DP05_0001E', 
'DP05_0002E', 
'DP05_0003E', 
'DP05_0004E', 
'DP05_0005E', 
'DP05_0006E', 
'DP05_0007E', 
'DP05_0008E', 
'DP05_0009E', 
'DP05_0010E', 
'DP05_0011E', 
'DP05_0012E', 
'DP05_0013E', 
'DP05_0014E', 
'DP05_0015E', 
'DP05_0016E', 
'DP05_0018E', 
'DP05_0019E', 
'DP05_0020E', 
'DP05_0021E', 
'DP05_0028E', 
'DP05_0030E', 
'DP05_0032E', 
'DP05_0033E', 
'DP05_0034E', 
'DP05_0039E', 
'DP05_0047E', 
'DP05_0052E', 
'DP05_0066E',  
'DP05_0081E']
# Will not work on Kaggle
#Download requested data from census bureau
C = get_census(file, year, geography, census_vars)
# For Kaggle
C = pd.read_pickle("../input/census-extract/census_extract.pkl")
# Rename columns
C = C.rename(columns={'DP05_0001E' : 'totpop', 
'DP05_0002E' : 'sex_male', 
'DP05_0003E' : 'sex_female', 
'DP05_0004E' : 'age_under_5', 
'DP05_0005E' : 'age_5_9', 
'DP05_0006E' : 'age_10_14', 
'DP05_0007E' : 'age_15_19', 
'DP05_0008E' : 'age_20_24', 
'DP05_0009E' : 'age_25_34', 
'DP05_0010E' : 'age_35_44', 
'DP05_0011E' : 'age_45_54', 
'DP05_0012E' : 'age_55_59', 
'DP05_0013E' : 'age_60_64', 
'DP05_0014E' : 'age_65_74', 
'DP05_0015E' : 'age_75_84', 
'DP05_0016E' : 'age_85_over', 
'DP05_0018E' : 'age_18_over', 
'DP05_0019E' : 'age_21_over', 
'DP05_0020E' : 'age_62_over', 
'DP05_0021E' : 'age_65_over', 
'DP05_0028E' : 'race_total', 
'DP05_0030E' : 'race_multi', 
'DP05_0032E' : 'race_white', 
'DP05_0033E' : 'race_black', 
'DP05_0034E' : 'race_native_amer', 
'DP05_0039E' : 'race_asian', 
'DP05_0047E' : 'race_pac_island', 
'DP05_0052E' : 'race_other', 
'DP05_0066E' : 'hispanic', 
'DP05_0081E' : 'housing_units'})

#C.info()
# Merge on intersection polygons with census data on NAME & tract_num
intersect_data = intersect_area.merge(C, how='left', left_on='NAME', right_on='tract_num')

#Sum polygon counts to LEA level
grouped = intersect_data.groupby("company")
LEA_census_data = grouped.sum().drop(columns=['inter_area', 'tract_area', 'LEA_area', 'prop_of_tract', 'prop_of_LEA'])
# Estimate population metric in each intersection polygon
def calc_LEA_percent(df, vars):
    for var in vars:
        df[var+"_p"] = 100 * df[var] / df['totpop']
varlist = list(LEA_census_data.columns)
calc_LEA_percent(LEA_census_data, varlist)
#LEA_census_data.info()
# Merge LEA-level census data to LEA geo data for mapping & analysis
LEA_shape_data = lea_shape.merge(LEA_census_data, how='left', on='company').drop(columns='totpop_p')
#LEA_shape_data.info()
# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Percent Asian Population for San Francisco Police Districts, ACS 5yr 2015", fontsize=15)

LEA_shape_data.plot(ax=ax, column='race_asian_p', cmap='BuPu', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(LEA_shape_data['race_asian_p']),  np.max(LEA_shape_data['race_asian_p'])
sm = plt.cm.ScalarMappable(cmap='BuPu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();
varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian','Black', 'Multi-race', 'Native American', 'Other', 'Pacific Island', 'White', 'Hispanic']

fig = plt.figure()
fig.set_size_inches(16, 20, forward=True)
fig.suptitle('Race/Ethnic Composition of LEA Districts', fontsize=16)

num=1

for var in varlist:
    plt.subplot(4, 2, num)
    
    # Choose the height of the bars
    height = LEA_shape_data[var]
 
    # Choose the names of the bars
    bars = list(LEA_shape_data['district'].unique())
    y_pos = np.arange(len(bars))
 
    # Create bars
    plt.bar(y_pos, height)
 
    # Create names on the x-axis
    plt.xticks(y_pos, bars, rotation=45, fontsize=8)
    plt.subplots_adjust(top=0.9, bottom=.1, hspace=.3)
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent")
 
    num+=1

# Show graphic
plt.show()
#Import LEA data file
file = '../input/data-science-for-good/cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv'
LEA_incidents_raw = pd.read_csv(file, skiprows=[1])
LEA_incidents_raw.info()
# Show indicent reasons
LEA_incidents_raw.INCIDENT_REASON.value_counts()
# Show most common incident dispositions
LEA_incidents_raw.DISPOSITION.value_counts().head(20)
from shapely.geometry import Point

# Create tuple of lat/long
LEA_incidents_raw['Coordinates'] = list(zip(LEA_incidents_raw.LOCATION_LONGITUDE, LEA_incidents_raw.LOCATION_LATITUDE))
# Convert to point
LEA_incidents_raw['Coordinates'] = LEA_incidents_raw['Coordinates'].apply(Point)
# Convert to geodataframe
LEA_incidents_shape = gpd.GeoDataFrame(LEA_incidents_raw, geometry='Coordinates')
# Set point geometry to same projection as Census files
LEA_incidents_shape.crs = {'init': 'epsg:4269'}
LEA_incidents_shape.head()
# Plot LEA districts, census tracts, and police incidents together

embez = LEA_incidents_shape[LEA_incidents_shape['INCIDENT_REASON']=="EMBEZZLEMENT"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

# set aspect to equal. This is done automatically
# when using *geopandas* plot on it's own, but not when
# working with pyplot directly.
ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Embezzlement Incidents in San Francisco", fontsize=15)

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
embez.plot(ax=ax, color='red', alpha=.5)
plt.show();
# Plot LEA districts, census tracts, and police incidents together

drug = LEA_incidents_shape[LEA_incidents_shape['INCIDENT_REASON']=="DRUG/NARCOTIC"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(10, 10, forward=True)

# set aspect to equal. This is done automatically
# when using *geopandas* plot on it's own, but not when
# working with pyplot directly.
ax.set_aspect('equal')
ax.set_xlim(-122.52, -122.36)
ax.set_ylim(37.7, 37.84)
ax.set_title("Drug/Narcotic Incidents in San Francisco", fontsize=15)

lea_shape.plot(ax=ax, color='white', edgecolor='black')
state_shape.plot(ax=ax, color='white', edgecolor='blue', alpha=.1)
drug.plot(ax=ax, color='green', alpha=.3)
plt.show();
#Will not work on Kaggle
#incidents_LEA_districts = gpd.sjoin(LEA_incidents_shape, lea_shape, how="inner", op='intersects')

# Read in result of spatial join
incidents_LEA_districts = pd.read_pickle("../input/spatial-join/incidents_LEA_districts.pkl")
incidents_LEA_districts.info()
incidents_LEA_districts['Embezzlements'] = incidents_LEA_districts['INCIDENT_REASON']=="EMBEZZLEMENT"
incidents_LEA_districts['Drugs'] = incidents_LEA_districts['INCIDENT_REASON']=="DRUG/NARCOTIC"
incidents_LEA_districts['Actions'] = incidents_LEA_districts['DISPOSITION']=="NONE" 
incidents_LEA_districts['EmbActions'] = incidents_LEA_districts.Embezzlements & incidents_LEA_districts.Actions
incidents_LEA_districts['DrugActions'] = incidents_LEA_districts.Drugs & incidents_LEA_districts.Actions
incidents_LEA_districts.head()
#Sum incident records to LEA level
grouped = incidents_LEA_districts.groupby('district')
incidents_grouped = grouped.sum().drop(columns=['index_right', 'shape_area', 'shape_le_1', 'shape_leng', 'INCIDENT_UNIQUE_IDENTIFIER', 'LOCATION_LONGITUDE', 'LOCATION_LATITUDE', 'Actions'])
#Create a few rates - remember Action means "NONE" before grouping
incidents_grouped['Emb_Action_Rate'] = 100-(100 * incidents_grouped.EmbActions/incidents_grouped.Embezzlements)
incidents_grouped['Drug_Action_Rate'] = 100-(100 * incidents_grouped.DrugActions/incidents_grouped.Drugs)
incidents_grouped.head()
LEA_integrated = LEA_shape_data.merge(incidents_grouped, how='left', on='district')
#LEA_integrated.info()
# Make choropleth map of population in LEA districts
fig, ax = plt.subplots()
fig.set_size_inches(11, 7, forward=True)

ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Percent of Drug Incidents In Which Police Action is Taken", fontsize=15)

LEA_integrated.plot(ax=ax, column='Drug_Action_Rate', cmap='Oranges', edgecolor='black')

# Create colorbar as a legend
vmin, vmax = np.min(LEA_integrated['Drug_Action_Rate']),  np.max(LEA_integrated['Drug_Action_Rate'])
sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)

plt.show();
varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian Percent','Black Percent', 'Multi-race Percent', 'Native American Percent', 'Other Percent', 'Pacific Island Percent', 'White Percent', 'Hispanic Percent']

fig = plt.figure()
fig.set_size_inches(12, 12, forward=True)
fig.suptitle('Race Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(3, 3, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drug_Action_Rate'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent Action Taken")
 
    num+=1

# Show graphic
plt.show()
varlist = ['race_asian_p','race_black_p', 'race_multi_p', 'race_native_amer_p', 'race_other_p', 'race_pac_island_p', 'race_white_p', 'hispanic_p']
var_labels = ['Asian Percent','Black Percent', 'Multi-race Percent', 'Native American Percent', 'Other Percent', 'Pacific Island Percent', 'White Percent', 'Hispanic Percent']

fig = plt.figure()
fig.set_size_inches(15, 12, forward=True)
fig.suptitle('Race Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(3, 3, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drug_Action_Rate'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Percent Action Taken")
 
    num+=1

# Show graphic
plt.show()
varlist = ['age_15_19_p','age_20_24_p']
var_labels = ['Percent 15-19','Percent 20-24']

fig = plt.figure()
fig.set_size_inches(14, 4, forward=True)
fig.suptitle('Age Composition and Action Taken in Drug Incidents in LEA Districts', fontsize=16)
num=1

for var in varlist:
    plt.subplot(1, 2, num)
    plot_var =  LEA_integrated[var]
    plt.scatter(plot_var, LEA_integrated['Drugs'])
    plt.tight_layout
    plt.title(var_labels[num-1])
    plt.ylabel("Number of Drug Incidents")
 
    num+=1

# Show graphic
plt.show()
