import folium

import pandas as pd
country_geo = '../input/python-folio-country-boundaries/world-countries.json'
# Read in the World Development Indicators Database

data = pd.read_csv('../input/world-development-indicators/Indicators.csv')

data.shape
data.head()
# select CO2 emissions for all countries in 2011

hist_indicator = 'CO2 emissions \(metric'

hist_year = 2011



mask1 = data['IndicatorName'].str.contains(hist_indicator) 

mask2 = data['Year'].isin([hist_year])



# apply our mask

stage = data[mask1 & mask2]

stage.head()
plot_data = stage[['CountryCode','Value']]

plot_data.head()
# label for the legend

hist_indicator = stage.iloc[0]['IndicatorName']
# Setup a folium map at a high-level zoom @Alok - what is the 100,0, doesn't seem like lat long

map = folium.Map(location=[100, 0], zoom_start=1.5)
# choropleth maps bind Pandas Data Frames and json geometries.  This allows us to quickly visualize data combinations

map.choropleth(geo_data=country_geo, data=plot_data,

             columns=['CountryCode', 'Value'],

             key_on='feature.id',

             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name=hist_indicator)
# Create Folium plot

map.save('plot_data.html')
# Import the Folium interactive html file

from IPython.display import HTML

HTML('<iframe src=plot_data.html width=700 height=450></iframe>')