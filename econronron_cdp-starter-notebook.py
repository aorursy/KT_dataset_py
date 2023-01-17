# standard libs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json

# plotting libs
import seaborn as sns

# geospatial libs
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon
import geopandas as gpd
import folium
import plotly.graph_objects as go
import plotly_express as px

# set in line plotly 
from plotly.offline import init_notebook_mode;
init_notebook_mode(connected=True)

print(os.getcwd())
# import corporate response data
cc_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')
ws_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2019_Full_Water_Security_Dataset.csv')
# import cities response df
cities_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")
# external data - import CDC social vulnerability index data - census tract level
svi_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/CDC Social Vulnerability Index 2018/SVI2018_US.csv")
# cities metadata - lat,lon locations for US cities
cities_meta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Simple Maps US Cities Data/uscities.csv")

# cities metadata - CDP metadata on organisation HQ cities
cities_cdpmeta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Locations of Corporations/NA_HQ_public_data.csv")
def list_dedupe(x):
    """
    Convert list to dict and back to list to dedupe
    
    Parameters
    ----------
    x: list
        Python list object
        
    Returns
    -------
    dictionary:
        dictionary object with duplicates removed
        
    """
    return list(dict.fromkeys(x))
cities_6_2 = cities_df[cities_df['Question Number'] == '6.2']\
    .rename(columns={'Organization': 'City'})

cities_6_2['Response Answer'] = cities_6_2['Response Answer'].fillna('No Response')

cities_6_2.head()
# state abbreviation dictionary
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# map dict to clean full state names to abbreviations
cities_cdpmeta_df['state'] = cities_cdpmeta_df['address_state'].map(us_state_abbrev)

# infill non-matched from dict
cities_cdpmeta_df['state'] = cities_cdpmeta_df['state'].fillna(cities_cdpmeta_df['address_state'])
cities_cdpmeta_df['state'] = cities_cdpmeta_df['state'].replace({'ALBERTA':'AB'})
cities_cdpmeta_df['address_city'] = cities_cdpmeta_df['address_city'].replace({'CALGARY':'Calgary'})
cities_cdpmeta_df= cities_cdpmeta_df.drop(columns=['address_state'])

# create joint city state variable
cities_cdpmeta_df['city_state'] = cities_cdpmeta_df['address_city'].str.cat(cities_cdpmeta_df['state'],sep=", ")

cities_cdpmeta_df
cities_count = cities_cdpmeta_df[['organization', 'address_city', 'state', 'city_state']]\
        .groupby(['address_city', 'state', 'city_state']).count().\
            sort_values(by = ['organization'],ascending = False)\
                .reset_index()\
                    .rename(columns={'organization' : 'num_orgs'})
cities_count.head()
# convert indexes to columns'
cities_count.reset_index(inplace=True)
cities_count = cities_count.rename(columns = {'index':'city_id'})
cities_df.reset_index(inplace=True)
cities_df = cities_df.rename(columns = {'index':'city_org_id'})

# convert id and city label columns into lists
city_id_no = list_dedupe(cities_count['city_id'].tolist())
city_name = list_dedupe(cities_count['address_city'].tolist())

city_org_id_no = list_dedupe(cities_df['city_org_id'].tolist())
city_org_name = list_dedupe(cities_df['Organization'].tolist())

# remove added index column in cities df
cities_df.drop('city_org_id', inplace=True, axis=1)
cities_count.drop('city_id', inplace=True, axis=1)

# zip to join the lists and dict function to convert into dicts
city_dict = dict(zip(city_id_no, city_name))
city_org_dict = dict(zip(city_org_id_no, city_org_name))
# compare dicts - matching when city name appears as a substring in the full city org name
city_names_df = pd.DataFrame(columns=['City ID No.','address_city', 'City Org ID No.','City Org', 'Match']) # initiate empty df

for ID, seq1 in city_dict.items():
    for ID2, seq2 in city_org_dict.items():
        m = re.search(seq1, seq2) # match string with regex search 
        if m:
            match = m.group()
            # Append rows in Empty Dataframe by adding dictionaries 
            city_names_df = city_names_df.append({'City ID No.': ID, 'address_city': seq1, 'City Org ID No.': ID2, 'City Org': seq2, 'Match' : match}, ignore_index=True)
            
# subset for city to city org name matches
city_names_df = city_names_df.loc[:,['address_city','City Org']]

city_names_df.head()
cities_count  = pd.merge(cities_count, city_names_df, on='address_city', how='left')
cities_count.head()
cities_6_2 = cities_6_2[['City', 'Response Answer']].rename(columns={'City' : 'City Org'})
cities_count = pd.merge(left=cities_count, right=cities_6_2, how='left', 
                        on ='City Org').rename(columns={'Response Answer' : 'Sustainability Project Collab.'})

cities_count['Sustainability Project Collab.'] = cities_count['Sustainability Project Collab.'].fillna('No Response')
cities_count_50 = cities_count.iloc[0:40,:]

plt.figure(figsize=(15,8))
ax = sns.barplot(
    x="city_state", y="num_orgs",
    hue = "Sustainability Project Collab.",
    data=cities_count_50 ,
    palette="OrRd_r"
)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='medium'  
)

# subset for lat, lng cities data
cities_meta_df = cities_meta_df[['city', 'state_id', 'lat','lng']].rename(columns={'city' : 'address_city', 'state_id' : 'state'})
cities_meta_df.head()
# join coordinates to cities count
cities_count = pd.merge(left=cities_count, right=cities_meta_df, how='left', on=['address_city', 'state'])

# convert text response to question 6.2 to an integar encoding 
resp_int_df = cities_count[["Sustainability Project Collab."]]
resp_int_df= resp_int_df.rename(columns={'Sustainability Project Collab.' : 'resp_int'})

labels = resp_int_df['resp_int'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
resp_int_df.replace({'resp_int': mapping},inplace=True)

resp_list = resp_int_df['resp_int'].tolist()
cities_count['resp_int'] = resp_list 
cities_count.head()
# plot spatial bubble map
cities_count['text'] = cities_count['address_city'] + '<br>Number of Orgs: ' + (cities_count['num_orgs']).astype(str) +\
    '<br>Sustainability Project Colloboration: ' + (cities_count['Sustainability Project Collab.']).astype(str)
limits = [(0,20),(21,40),(41,60),(61,80),(81,100)]
cities = []
scale = 5

fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = cities_count['lng'],
        lat = cities_count['lat'],
        text = cities_count['text'],
        marker = dict(
            size = cities_count['num_orgs']*scale,
            color = cities_count['resp_int'],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = '2019 CDP Climate Change Corporate Responders (Public) by City',
        showlegend = False,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()
cc_2_4a = cc_df[cc_df['question_number'] == 'C2.4a']
cities_cdpmeta_join = cities_cdpmeta_df[["account_number", 'survey_year', 'address_city']]
cc_2_4a = pd.merge(left=cc_2_4a, right=cities_cdpmeta_join,  left_on=['account_number','survey_year'], right_on = ['account_number','survey_year'])
cc_nyc = cc_2_4a[(cc_2_4a['address_city'] =='New York')]
cities_6_2['City Org'] = cities_6_2['City Org'].replace({'New York City':'New York'})
cc_nyc = pd.merge(left=cc_nyc, right= cities_6_2,  left_on=['address_city'], right_on = ['City Org']).rename(columns={'Response Answer' : 'sustain_collab'})
cc_nyc.head()
nyc_svi_df = svi_df[svi_df['STCNTY'].isin([36005, 36047, 36061, 36081, 36085])]
nyc_svi_df['City'] = 'New York'
print(nyc_svi_df.shape)
nyc_svi_df.head()
# import shapefile of NYC census tracts
geodf = gpd.read_file('../input/cdp-unlocking-climate-solutions/Supplementary Data/NYC CDP Census Tract Shapefiles/nyu_2451_34505.shp')

# join geospatial data to SVI unemployment rates ('E_UNEMP')
gdf_join = geodf[['tractid', 'geometry']].to_crs('+proj=robin')
nyc_join =  nyc_svi_df[['E_UNEMP', 'FIPS']]
gdf_join["tractid"] = pd.to_numeric(geodf["tractid"])
gdf_nyc = pd.merge(left=gdf_join, right=nyc_join, how='left', left_on='tractid', right_on = 'FIPS')
gdf_nyc.head()
# plot unemployment rate variation across NYC
colors = 5
cmap = 'RdPu'
figsize = (16, 10)
ax = gdf_nyc.dropna().plot(column='E_UNEMP', cmap=cmap, figsize=figsize, scheme='equal_interval', k=colors, legend=True)
ax.set_title('CDV SVI Civilian (age 16+) unemployed estimate by NYC Census Tract', fontdict={'fontsize': 16}, loc='center')
ax.get_legend().set_bbox_to_anchor((.40, .8))
del cc_df 
del cities_df 
del svi_df
del cities_meta_df 
del cities_cdpmeta_df
del cities_6_2
del cities_count
del cc_2_4a
# subset for Bronx
bb_df = nyc_svi_df[(nyc_svi_df.COUNTY =='Bronx')]

# join to city and climate change response data
print(cc_nyc.shape)
cc_nyc = cc_nyc.rename(columns={'City Org' : 'City'})
nyc_df = pd.merge(cc_nyc,bb_df,on='City',how='outer')
print(nyc_df.shape)
nyc_df.head()
ws_df_4_1c = ws_df[ws_df['question_number'] == 'W4.1c']
ws_df_4_1c = ws_df_4_1c[ws_df_4_1c['response_value'].notnull()]
ws_df_4_1c.head()         
# pivot data
ws_df_4_1c_wide = ws_df_4_1c.pivot_table(index=['account_number', 'organization', 'row_number'],
                                     columns='column_name', 
                                     values='response_value',
                                     aggfunc=lambda x: ' '.join(x)).reset_index()
# identify orgs with facilities within the Hudson river basin
ws_df_4_1c_wide = ws_df_4_1c_wide[ws_df_4_1c_wide['W4.1c_C2River basin'].str.contains('Hudson', na=False)]
ws_df_4_1c_wide.head()
ws_df.head()
