# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from pprint import pprint

# Thanks to paultimothymooney for outlining how to use bq_helper to query this dataset
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
usfs = BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")

# Need SQLite for forest fire dataset
import sqlite3
conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')
query1 = """
SELECT
    species_common_name,
    total_height,
    current_diameter,
    measurement_year,
    plot_county_code
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    AND total_height > 0
;        """
response1 = usfs.query_to_pandas_safe(query1, max_gb_scanned=10)

# Group together height and diameter measurements for each species taken each year
height_lst = response1.groupby(['species_common_name', 'measurement_year']).agg({'current_diameter': 'mean'})

# Remove tree types that have at least 12 measurements, and transpose so subplots groups species and not year
height_df = height_lst.unstack().dropna(axis='index', thresh=12)

# Rename columns to remove total_height label
height_df.columns = height_df.columns.map(lambda t: t[1])
_ = height_df.transpose().plot.bar(subplots=True, layout=(40,2), figsize=(20, 100))
query2 = """
SELECT
    AVG(current_diameter) as avg_dia,
    measurement_year,
    plot_county_code
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    AND total_height > 0
GROUP BY
    plot_county_code,
    measurement_year
;        """
response2 = usfs.query_to_pandas_safe(query2, max_gb_scanned=10)
# Turn the dataframe into a pivot table with the avg_dia as the values inside
pivoted = response2.pivot(index='measurement_year', columns='plot_county_code')
# Remove any year that less than 80% the counties were measured
pivoted.dropna(inplace=True, thresh=len(pivoted.columns.values)*0.8)
# Drop counties missing measurements
pivoted.dropna(axis='columns', inplace=True)
# Rename columns to remove avg_dia from header
pivoted.columns = [code for _, code in pivoted.columns.values]

# Create dictionaries to convert county code to county name and vice versa
wa_counties = ['Adams', 'Asotin', 'Benton', 'Chelan', 'Clallam', 'Clark', 'Columbia', 'Cowlitz', 'Douglas', 'Ferry', 'Franklin'
               , 'Garfield', 'Grant', 'Grays Harbor', 'Island', 'Jefferson', 'King', 'Kitsap', 'Kittitas', 'Klickitat'
               , 'Lewis', 'Lincoln', 'Mason', 'Okanogan', 'Pacific', 'Pend Oreille', 'Pierce', 'San Juan', 'Skagit', 'Skamania'
               , 'Snohomish', 'Spokane', 'Stevens', 'Thurston', 'Wahkiakum', 'Walla Walla', 'Whatcom', 'Whitman', 'Yakima']
wa_county_codes = ['001', '003', '005', '007', '009', '011', '013', '015', '017', '019', '021', '023', '025'
                   , '027', '029', '031', '033', '035', '037', '039', '041', '043', '045', '047', '049', '051'
                   , '053', '055', '057', '059', '061', '063', '065', '067', '069', '071', '073', '075', '077']
# Get county name given code
wa_county_dict = {int(code): name.lower() for code, name in zip(wa_county_codes, wa_counties)}
# Get code given county name
wa_code_dict = {name.lower(): int(code) for code, name in zip(wa_county_codes, wa_counties)}

# Create the correlation table with the headers as the county codes
df_corr = pivoted.corr()

# Change the column headers to the country names for printing to make the pivot table easier to read
pivoted_named = pivoted.copy()
pivoted_named.columns = [wa_county_dict[code] for code in pivoted.columns.values]
df_corr_named = pivoted_named.corr()

# Colour the cells green when strongly correlated and red when not
import seaborn as sns
df_corr_color = df_corr_named.style.background_gradient(cmap='RdYlGn')
df_corr_color

# TODO: Could look into how correlated height and diameters are between counties as a way to determine that they have similar tree types
df_diff_named = pivoted_named.diff()
df_corr2 = df_diff_named['columbia'].rolling(window=4).corr(other=df_diff_named['kitsap'])
df_std = df_corr2.rolling(window=2).std()
fig, ax = plt.subplots()
df_corr2.plot(title='Diameter of Trees in Columbia vs. Kitsap Counties', x='Year', legend=True, ax=ax)
df_std.plot(legend=True, ax=ax)
_ = ax.legend(['corr', 'std'])
# Check if every time this STD is above 0.7 if a fire occurred in either county by looking at a couple examples
# First field is year, next 2 are the counties that it seems that a fire occurred
pred_fire_years = []
thresh = 0.7
# Get the difference in diameter each year in each county
df_diff = pivoted.diff()
years=df_diff.index.tolist()

for county1 in df_diff.columns.values:
    for county2 in df_diff.columns.values:
        if county1 == county2:
            continue
        # Get the correlation between each pair of counties
        df_corr2 = df_diff[county1].rolling(window=4).corr(other=df_diff[county2])
        # Find the std of the correlation like before
        lst_std = df_corr2.rolling(window=2).std()
        # If any are over 0.7 append them to the list
        i = np.where(lst_std >= thresh)[0]
        if len(i):
            [pred_fire_years.append((years[index], county1, county2)) for index in i]
pprint(pred_fire_years[:5])
# Testing the accuracy of using the std to determine if a fire occurred in that county
fire_by_year = pd.read_sql_query("SELECT group_concat(COUNTY) as counties, FIRE_YEAR FROM fires \
                                WHERE State='WA' AND FIRE_SIZE_CLASS IN ('C', 'D', 'E', 'F', 'G') AND COUNTY IS NOT NULL\
                                GROUP BY FIRE_YEAR;", conn)

# Convert the dataframe of counties to a dict of a list all the county codes that had a fire in each FIRE_YEAR
fire_county_by_year_dict = {}
fipa_counties = set(wa_county_dict.values())
for count, year in enumerate(fire_by_year['FIRE_YEAR']):
    fire_county_by_year_dict[year] = []
    for counter, county in enumerate(fire_by_year['counties'][count].split(',')):
        try:
            # Some rows are the correct number so just convert from str to int and insert
            fire_county_by_year_dict[year].append(int(county))
        except ValueError:
            # The rest are named, convert these to the correct number then insert
            county_name = county.split(' ')[0].lower()
            try:
                # If county name is valid this will append its county code to the dict
                fire_county_by_year_dict[year].append(wa_code_dict[county_name])
            except KeyError:
                # County name is invalid, skip
                continue

# Initialize dict of counts of times that the prediction was correct in a county for that year
verified_fire_years = {'_'.join([str(year), str(county1)]):0 for year, county1, county2 in pred_fire_years}
[verified_fire_years.update({'_'.join([str(year), str(county2)]):0}) for year, county1, county2 in pred_fire_years]

for year, county1, county2 in pred_fire_years:
    if year > 2015 or year < 1992:
        # Forest Fire database only goes from 1992 to 2015
        continue
    counties_with_fire_that_year = fire_county_by_year_dict[year]
    label1 = '_'.join([str(year), str(county1)])
    label2 = '_'.join([str(year), str(county2)])
    # If a predicted fire is found in the list of fires that happened that year then add 1 to the count
    # since multiple pairs of counties may have predicted a fire
    if county1 in counties_with_fire_that_year:
        if label1 in verified_fire_years.keys():
            verified_fire_years[label1] += 1
        else:
            verified_fire_years[label1] = 1
    if county2 in counties_with_fire_that_year:
        if label2 in verified_fire_years.keys():
            verified_fire_years[label2] += 1
        else:
            verified_fire_years[label2] = 1
    # Means that std predicted a fire that never happened, subtract a count
    else:
        if label1 in verified_fire_years.keys():
            verified_fire_years[label1] -= 1
        else:
            verified_fire_years[label1] = 0
        if label2 in verified_fire_years.keys():
            verified_fire_years[label2] -= 1
        else:
            verified_fire_years[label2] = 0
        verified_fire_years[label2] = 0

accuracy = sum(verified_fire_years.values())/len(pred_fire_years)
print(accuracy * 100)
# Now to try and visualize this impact, first plot a heat map where the intensity of the colour in each count shows
# the relative number of forest fires in that county compared to the max in Washington

# Get the locations of all forest fires in Washington state that exceeded 100 acres in size(class F-G)
df = pd.read_sql_query("SELECT LATITUDE, LONGITUDE, COUNTY FROM fires \
                        WHERE State='WA' AND FIRE_SIZE_CLASS IN ('C', 'D', 'E', 'F', 'G') AND COUNTY IS NOT NULL\
                        AND FIRE_YEAR = 2010;", conn)
ffire_lats = df['LATITUDE']
ffire_lons = df['LONGITUDE']
ffire_county = df['COUNTY'].tolist()

for fipa_county, fipa_code in zip(wa_counties, wa_county_codes):
    for counter, county in enumerate(ffire_county):
        try:
            # Some rows are the correct number so just convert from str to int and insert
            ffire_county[counter] = int(county)
        except ValueError:
            # The rest are named, convert these to the correct number then insert
            if fipa_county.lower() in county.lower():
                ffire_county[counter] = int(fipa_code)

# bounding box of Washington State
bbox_ll = [45.543541, -124.848974]
bbox_ur = [49.002431, -116.915580]

# geographical center of united states
lat_0 = 39.833333
lon_0 = -98.583333


from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from collections import Counter

ffire_counts = Counter(ffire_county)

fig = plt.figure(figsize=(12,8))

# create polar stereographic Basemap instance.
Map = Basemap(projection='merc',
            lon_0=lon_0,lat_0=90.,lat_ts=lat_0,
            llcrnrlat=bbox_ll[0],urcrnrlat=bbox_ur[0],
            llcrnrlon=bbox_ll[1],urcrnrlon=bbox_ur[1],
            rsphere=6371200.,resolution='l',area_thresh=10)

# draw coastlines, county, state and country boundaries and the edge of the map
Map.drawcoastlines()
Map.drawstates()
Map.drawcountries()
Map.drawcounties()

colorVotes = plt.get_cmap('Greens')
colors = []
patches = []

# get all the county names for gathering the shapefile attributes to find within the counties list of dicts
county_names = []
for county_dict in Map.counties_info:
    county_names.append((int(county_dict['COUNTY_FIP']), county_dict['STATE']))
ax = plt.gca()

# Get the Polygon shape file from the list of counties and fill it in with a colour corresponding to the relative
# number of forest fires in the county
for county, value in ffire_counts.items():
    seg = Map.counties[county_names.index((county, 'WA'))]
    color = colorVotes(value/float(max(ffire_counts.values())))
    colors.append(color) # add colors to colors list
    poly = Polygon(seg, facecolor=color, edgecolor=color)
    patches.append(poly) #add polygon to patches list
    ax.add_patch(poly)

# Some counties will not color in, possibly due to bad or missing shape file for that county or they didn't 
# have any measurements taken in them in this dataset
p = PatchCollection(patches, cmap=colorVotes)
p.set_array(np.array(colors))
cb = fig.colorbar(p)
# draw points where forest fires occurred
x,y = Map(ffire_lons.values, ffire_lats.values)
Map.plot(x, y, 'ro', markersize=4)

plt.show()
# Visualize the change in tree diameter in each country with the forest fires in each county plotted for that year

diff_dia_dict = {year: diff for year, diff in zip(df_diff.index.tolist(), df_diff.values.tolist())}
county_codes = df_diff.columns.values.tolist()

fig = plt.figure(figsize=(12, 8))
# create polar stereographic BaseMap2 instance.
Map2 = Basemap(projection='merc',
            lon_0=lon_0,lat_0=90.,lat_ts=lat_0,
            llcrnrlat=bbox_ll[0],urcrnrlat=bbox_ur[0],
            llcrnrlon=bbox_ll[1],urcrnrlon=bbox_ur[1],
            rsphere=6371200.,resolution='l',area_thresh=10)

# draw coastlines, county, state and country boundaries and the edge of the Map2
Map2.drawcoastlines()
Map2.drawstates()
Map2.drawcountries()
Map2.drawcounties(linewidth=0.3)
#print(Map2.counties_info)

colorVotes = plt.get_cmap('RdYlGn')
colors = []
patches = []

# get all the county names for gathering the shapefile attributes to find within the counties list of dicts
county_names = []
for county_dict in Map2.counties_info:
    county_names.append((int(county_dict['COUNTY_FIP']), county_dict['STATE']))
ax = plt.gca()
#fig = plt.figure(figsize=(6,4))
#fig2 = plt.subplots()

# draw points where forest fires occurred
x,y = Map2(ffire_lons.values, ffire_lats.values)
point = Map2.plot(x, y, 'ro', markersize=2)[0]

print(diff_dia_dict[2010])

for county, value in zip(county_codes, diff_dia_dict[2011]):
    seg = Map2.counties[county_names.index((county, 'WA'))]
    color = colorVotes(value)
    colors.append(color) # add colors to colors list
    poly = Polygon(seg, facecolor=color, edgecolor=color)
    patches.append(poly) #add polygon to patches list
    ax.add_patch(poly)
    plt.title(2010)

p = PatchCollection(patches, cmap=colorVotes)
p.set_array(np.array(colors))
cb = Map2.colorbar(p, ticks=[0,0.5,1])
cb.ax.set_yticklabels([str(min(diff_dia_dict[2010])), str(sum(diff_dia_dict[2010])/len(diff_dia_dict[2010])), str(max(diff_dia_dict[2010]))])
cb.ax.set_ylabel('Change in Tree Diameter')

plt.show()