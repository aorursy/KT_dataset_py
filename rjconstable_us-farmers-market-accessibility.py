# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

import seaborn as sns
markets = pd.read_csv('../input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')

counties = pd.read_csv('../input/farmers-markets-in-the-united-states/wiki_county_info.csv')
pd.set_option('display.max_columns', None)

markets.head(5)
markets.columns
counties.head(5)
# groupby county and groupby state to get markets/county, markets/state

market_per_county = markets.groupby('County')

market_per_state = markets.groupby('State')
# count the number of markets per state

market_per_state = market_per_state['MarketName'].count()

market_per_state.sort_values(inplace = True)
ax = market_per_state.plot.bar(figsize = (10, 6))

ax.set_ylabel('Market Count')

ax.set_title('Market Count (States)');
counties.head()
counties.columns
counties.dropna(inplace = True)
# lets format the numerical wealth metrics so they can be used

wealth_metrics = ['per capita income','median household income', 'median family income']

for i in wealth_metrics:

    counties[i] = counties[i].str.strip('$,')

    counties = counties.replace(',','', regex=True)

    counties[i] = counties[i].astype('int32')
counties.head()
# we can now groupby state and average over the wealth metrics

counties_state_gb = counties.groupby(['State'])[wealth_metrics].mean().reset_index()

counties_state_gb.sort_values(by = 'per capita income', inplace = True)
counties_state_gb.head()
# In the interests of brevity lets focus on per capita income for now

ax = counties_state_gb[['State', 'per capita income']].plot.bar(x = 'State', figsize = (10, 6))

ax.set_ylabel('per capita income')

ax.set_title('Per Capita Income (States)');
# combine the markets and counties tables and plot on one chart

combined_state_df = pd.merge(market_per_state, counties_state_gb, how = 'left', on = 'State')

combined_state_df.sort_values(by = 'per capita income', inplace = True)
combined_state_df.head()
# sum the number of househodls per state and use this to calculate the markets per capita for each state

counties['population'] = counties['population'].astype('int32')

counties_sum = counties.groupby('State').sum().reset_index()
counties_sum.head()
# we don't need the sum of the other columns so isolate the state and population columns

counties_sum = counties_sum[['State', 'population']]
combined_state_df.head()
counties_sum.head()
combined_state_pop_df = pd.merge(combined_state_df, counties_sum, how = 'right', on = 'State')

combined_state_pop_df.head(5)
combined_state_pop_df['market_per_capita'] = combined_state_pop_df['MarketName'] / combined_state_pop_df['population']

combined_state_pop_df.head(5)
# min max scaling to get the no. of markets and per capita income on a comparable scale for plotting

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

x = combined_state_pop_df[['market_per_capita']].values.astype(float)

x_scaled = min_max_scaler.fit_transform(x)

combined_state_pop_df['norm_mpc'] = x_scaled



min_max_scaler = preprocessing.MinMaxScaler()

x = combined_state_pop_df[['per capita income']].values.astype(float)

x_scaled = min_max_scaler.fit_transform(x)

combined_state_pop_df['norm_pci'] = x_scaled
combined_state_pop_df.head()
# By eye, is there any correlation between markets per capita and income per capita, at the state level? Answer: maybe

f, ax = plt.subplots(figsize=(10, 5))

plt.xticks(rotation=90, fontsize=10)

plt.ylabel('Normalised Value')

plt.bar(height="norm_pci", x="State", data=combined_state_pop_df, label="Per Capita Income", color="lightgreen", alpha = 0.5);

plt.bar(height="norm_mpc", x="State", data=combined_state_pop_df, label="Per Capita Markets", color="black", alpha = 0.5);

plt.title('Per Capita Income vs Per Capita Markets (States)')

plt.legend();
# checking for correlation between markets per capita and income per capita at a state level

import scipy.stats as stats

g = sns.jointplot(data = combined_state_pop_df, x = "per capita income", y = "market_per_capita", kind = 'reg')

g.annotate(stats.pearsonr);
# we've got two county columns in the markets and counties dataframe, lets use sets and an intersection operation to find those county values common to both

unique_counties_markets = set(markets['County'].unique())

unique_counties_counties = set(counties['county'].unique())

shared_counties = unique_counties_markets.intersection(unique_counties_counties)
len(counties)
len(shared_counties)
markets['county'] = markets['County']
# use the isn() operator to subset each dataframe based on shared_counties set membership and then merge them

markets_shared = markets.loc[markets['county'].isin(shared_counties)]

counties_shared = counties.loc[counties['county'].isin(shared_counties)]

combined_county = pd.merge(markets_shared, counties_shared, how = 'left', on = 'county')
combined_county['county'].value_counts()
combined_county.columns
# state to state2letter. Thanks to https://gist.github.com/rogerallen/1583593

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
State_2_letter = []

for i in combined_county['State_x']:

    State_2_letter.append(us_state_abbrev.get(i))

combined_county['State_2_letter'] = State_2_letter
combined_county['State_2L_county'] = combined_county['State_2_letter'] + ', ' + combined_county['county']
combined_county['State_2L_county'].head()
combined_county['State_2L_county'].unique()
# calculate markets per capita (county)

market_per_county = combined_county.groupby('State_2L_county', as_index=False)

market_per_county = market_per_county['MarketName'].count()

market_per_county = pd.DataFrame(market_per_county)
market_per_county.columns
market_per_county.rename(columns = {'State_2L_county':'county'}, inplace=True)

combined_county.drop(['county'], axis=1, inplace=True)

combined_county.rename(columns = {'State_2L_county':'county'}, inplace=True)
combined_county.shape
combined_county.columns
# sum the populations over the counties and get a mean of each wealth metric over the counties in separate dataframes

combined_county_population = combined_county.groupby(['county'])['population'].sum().reset_index()

combined_county_pci = combined_county.groupby(['county'])[wealth_metrics].mean().reset_index()
combined_county_population.head()
combined_county_pci.head()
market_per_county.columns
#combine the dataframes and calculate the markets per capita

county_markets_pop = pd.merge(market_per_county, combined_county_population, how = 'left', on = 'county')

county_markets_pop_pci = pd.merge(county_markets_pop, combined_county_pci, how = 'left', on = 'county')

county_markets_pop_pci['markets_pc'] = county_markets_pop_pci['MarketName'] / county_markets_pop_pci['population']
county_markets_pop_pci
# scaling again for our simple barplot comparison of counties sorted by per capita income 

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

x = county_markets_pop_pci[['markets_pc']].values.astype(float)

x_scaled = min_max_scaler.fit_transform(x)

county_markets_pop_pci['norm_mpc'] = x_scaled



min_max_scaler = preprocessing.MinMaxScaler()

x = county_markets_pop_pci[['per capita income']].values.astype(float)

x_scaled = min_max_scaler.fit_transform(x)

county_markets_pop_pci['norm_pci'] = x_scaled



county_markets_pop_pci.sort_values(by = 'per capita income', inplace = True)
# many counties in puerto rico are topping the list ;)

county_markets_pop_pci.head(10)
# Too many counties to show on the x axis conveniently on a non-interactive plot but we can visualise the relationship between per capita income and per capita markets at the county level

f, ax = plt.subplots(figsize=(10, 5))

plt.xticks(rotation=90, fontsize=10)

plt.ylabel('Normalised Value')

plt.xlabel('Counties')

plt.bar(height="norm_pci", x="county", data=county_markets_pop_pci, label="Per Capita Income", color="lightgreen", alpha = 0.5);

plt.bar(height="norm_mpc", x="county", data=county_markets_pop_pci, label="Per Capita Markets", color="black", alpha = 0.5);

plt.title('Per Capita Income vs Per Capita Markets (Counties)')

plt.legend();
g = sns.jointplot(data = county_markets_pop_pci, x = "per capita income", y = "markets_pc", kind = 'reg')

g.annotate(stats.pearsonr);
combined_state_pop_df.head()
combined_state_pop_df.rename(columns={'MarketName':'Market_Count'}, inplace=True)
# Add some extra columns to the wealth_metrics list to use for specify data for the correlation matrices

numericals = ['Market_Count', 'per capita income',

       'median household income', 'median family income', 'population', 'market_per_capita']
from string import ascii_letters



def plot_corr_matrix(df, columns, plot_title):

    sns.set(style="white")

    correlations = df[columns].corr()

    mask = np.triu(np.ones_like(correlations, dtype=np.bool))

    f, ax = plt.subplots(figsize=(10, 6))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax = sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5, "label": "Pearsons r"})

    ax.set_title(plot_title)

    plt.show()



plot_corr_matrix(combined_state_pop_df, numericals, 'State Level - Numerical Features vs Markets')
county_markets_pop_pci.rename(columns={'markets_pc':'market_per_capita'}, inplace=True)

county_markets_pop_pci.rename(columns={'MarketName':'Market_Count'}, inplace=True)
# in the absence of consistent column names....we must specify what things are in the counties df

# numericals_counties = ['Market_Count', 'per capita income',

#        'median household income', 'median family income', 'population', 'markets_pc']
# my dataframe names could be clearer...

county_markets_pop_pci.head()
plot_corr_matrix(county_markets_pop_pci, numericals, 'County Level - Numerical Features vs Markets')
# plotly chloropleth tutorial here: https://plotly.com/python/mapbox-county-choropleth/

from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties_json = json.load(response)  # load the geographical data for US counties
# load the FIPS county codes 

!curl -O https://coastwatch.pfeg.noaa.gov/erddap/convert/fipscounty.csv
fips_county = pd.read_csv('./fipscounty.csv')
fips_county.head()
fips_county.shape
# Get rid of the state only rows with a lambda function

fips_county_names = fips_county[fips_county['Name'].apply(lambda x: len(x) > 2)]
fips_county_names.shape
fips_county_names.head()
county_markets_pop_pci.head()
fips_county_names.rename(columns={'Name':'county'}, inplace=True)
fips_county_names['county'] = fips_county_names['county'].astype('str')

fips_county_names['FIPS'] = fips_county_names['FIPS'].astype('str');
county_markets_pop_pci['county'] = county_markets_pop_pci['county'].astype('str')

county_markets_pop_pci['county'] = county_markets_pop_pci['county'].str.strip()
# make a single dataframe with the fips codes and markets per capita

combo_FIPS_markets = pd.merge(fips_county_names, county_markets_pop_pci, how = 'inner', on = 'county')
# FIPS are 5 digit codes but the 0 are missing from the first 10000 in the data I've found here. So lets add some zeros

def add_zero(df):

    for i,j in enumerate(df['FIPS']):

        if len(j) == 4:

            df['FIPS'][i] = '0'+ j

        else:

            continue

    return df
add_zero(combo_FIPS_markets)
#combo_FIPS_markets['FIPS'].loc[combo_FIPS_markets['county'].str.contains('CA,')]
counties_copy = counties

State_2_letter = []

for i in counties_copy['State']:

    State_2_letter.append(us_state_abbrev.get(i))

counties_copy['State_2_letter'] = State_2_letter
counties_copy['State_2L_county'] = counties_copy['State_2_letter'] + ', ' + counties_copy['county']
counties_copy.drop(['county'], axis=1, inplace=True)

counties_copy.rename(columns = {'State_2L_county':'county'}, inplace=True)
# Now we have a dataframe with almost complete per capita income data for each US county

county_pc_FIPS = pd.merge(fips_county_names, counties_copy, how = 'inner', on = 'county')
add_zero(county_pc_FIPS)
import plotly.express as px



fig = px.choropleth_mapbox(county_pc_FIPS, geojson=counties_json, locations='FIPS', color='per capita income',

                           color_continuous_scale="Viridis",

                           range_color=(0, max(county_pc_FIPS['per capita income'])),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.5,

                           labels={'per capita income':'per capita income'}

                          )



#fig.add_trace(px.scatter_mapbox(markets, lat=markets['x'], lon=markets['y']))



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, title = 'Per capita income in US counties')

# fig.update_layout()

fig.show()
import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergeo(

        lon = markets['x'],

        lat = markets['y'],

        mode = 'markers',

        marker_color = 'blue',

        geojson = counties_json

        ))

fig.update_layout(geo_scope='usa', title = 'Farmers Market Locations')

fig.show()
fig = px.choropleth_mapbox(combo_FIPS_markets, geojson=counties_json, locations='FIPS', color='market_per_capita',

                           color_continuous_scale="picnic",

                           range_color=(0, max(combo_FIPS_markets['market_per_capita'])),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.5,

                           labels={'market_per_capita':'markets per capita'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()