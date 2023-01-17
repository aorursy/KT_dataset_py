# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import folium

import json

import pandas as pd

with open('/kaggle/input/world-countries/world-countries.json') as data_file:

   country_geo = json.load(data_file)
data = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')

data.shape
data.head()
countries = data['CountryName'].unique().tolist()

indicators = data['IndicatorName'].unique().tolist()

print(len(countries))

print(len(indicators))
data['IndicatorName'][:25]
hist_indicator =  'Life expectancy at birth'

hist_year = 2013

mask1 = data['IndicatorName'].str.contains(hist_indicator) 

mask2 = data['Year'].isin([hist_year])

# apply our mask

stage = data[mask1 & mask2]

stage.head()
data_to_plot = stage[['CountryCode','Value']]

data_to_plot.head()
hist_indicator = stage.iloc[0]['IndicatorName']
map = folium.Map(location=[100, 0], zoom_start=1.5)

map.choropleth(geo_data=country_geo, data=data_to_plot,

             columns=['CountryCode', 'Value'],

             key_on='feature.id',

             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,

             legend_name=hist_indicator)
# Create Folium plot

x = map.save('plot_data.html')

# Import the Folium interactive html file

from IPython.display import IFrame

IFrame(src= './plot_data.html', width=1000 ,height=450)
