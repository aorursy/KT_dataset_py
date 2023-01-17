%matplotlib inline

# Data Processing

import pandas as pd

import numpy as np



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set_palette('husl')



# Special data visualization

import missingno as msno # check missing value

from wordcloud import WordCloud # wordcloud

import plotly.graph_objects as go



# Geographic data visualization

import pycountry

import plotly.express as px



# Check file list

import os

print(os.listdir('../input/wine-reviews'))
wines = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')

wines.describe(include = 'all')
wines.head()
msno.matrix(wines, color = (255/255, 38/255, 90/255))
wines.dropna(subset = ['country', 'price', 'variety'], inplace = True)



# Visualizing missing values after deleting

msno.matrix(wines, color = (57/255, 194/255, 110/255))
%%time

fig, ax = plt.subplots(1, 2, figsize=(16, 32))

wordcloud_description = WordCloud(background_color='white',width=800, height=800).generate(' '.join(wines['description']))

wordcloud_variety = WordCloud(background_color='white',width=800, height=800).generate(' '.join(wines['variety']))

ax[1].imshow(wordcloud_variety, interpolation='bilinear')

ax[1].set_title("Wines variety")

ax[1].axis('off')

ax[1].margins(x=0, y=0)

ax[0].imshow(wordcloud_description, interpolation='bilinear')

ax[0].set_title("Wines description")

ax[0].axis('off')

ax[0].margins(x=0, y=0)

plt.show()
fig, axs = plt.subplots(ncols = 2, figsize = (25, 10))

sns.set_style("whitegrid")

sns.kdeplot(wines['price'], shade = True, legend = False, ax = axs[0],  color = 'orange')

axs[0].set_title("Wine prices distribution")

sns.kdeplot(wines['points'], shade = True, legend = False, ax = axs[1], color = 'green')

axs[1].set_title("Wine points distribution")

plt.plot()
fig = plt.figure(figsize = (20, 10), dpi = 72)

sns.regplot(x = wines['points'], y = wines['price'], color = 'red')
wines_countries_counts = wines['country'].value_counts()

wines_countries_points = pd.Series(index = wines_countries_counts.index)

for country in wines['country'].unique():

    country_name = wines[wines['country'] == country]

    wines_countries_points.loc[country] = country_name['points'].mean()

    

wines_countries = pd.concat({'total': wines_countries_counts,'points': wines_countries_points}, axis = 1)



# Converting country names to iso_alpha_3

countries = {'US': 'USA', 'England': 'GBR', 'Moldova': 'MDA', 'Macedonia': 'MKD', 'Czech Republic': 'CZE'}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

for country, row in wines_countries.iterrows():

    wines_countries.loc[country, 'iso_alpha'] = countries[country]

wines_countries.head()
fig = px.choropleth(wines_countries, locations="iso_alpha",

                    color="total",

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title='Amount of wines by country')

fig.show()
fig = px.choropleth(wines_countries, locations="iso_alpha",

                    color="points",

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title='Average wine points by country')

fig.show()
us_iso = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

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

    'Palau': 'PW',

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

    'Wyoming': 'WY',

}
us_wines_counts = wines.loc[wines['country'] == 'US', 'province'].value_counts()

us_wines_points = pd.Series(index = us_wines_counts.index)

states = wines.loc[wines['country'] == 'US', 'province'].unique()

for state in states:

    state_name = wines[wines['province'] == state]

    us_wines_points.loc[state] = state_name['points'].mean()



us = pd.concat({'total': us_wines_counts,'points': us_wines_points}, axis = 1)

us.drop(['America', 'Washington-Oregon'], inplace = True)



# Converting state codes

states = {}

for state in us_iso:

    states[state] = us_iso[state]

    

for state, row in us.iterrows():

    us.loc[state, 'iso_alpha'] = states[state]

    

us.head()
fig = go.Figure(data=go.Choropleth(

    locations=us['iso_alpha'], # Spatial coordinates

    z = us['total'], # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Blues',

    colorbar_title = "Total",

))



fig.update_layout(

    title_text = 'US total wines by state',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
fig = go.Figure(data=go.Choropleth(

    locations=us['iso_alpha'], # Spatial coordinates

    z = us['points'], # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Reds',

    colorbar_title = "Points",

))



fig.update_layout(

    title_text = 'US Average wine points by State',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
labels = wines_countries.index

values = wines_countries['total']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
tasters = wines['taster_name'].value_counts()

fig = go.Figure(data=[go.Pie(labels=tasters.index, values=tasters, hole=.5)])

fig.show()