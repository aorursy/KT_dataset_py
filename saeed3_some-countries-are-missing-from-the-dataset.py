# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')



print(train_df.shape)

print(test_df.shape)
print(train_df.columns)

print(train_df.head(5))

print(set(train_df['Country_Region'].tolist()))
grouped = train_df.groupby('Date')

grouped = grouped['ConfirmedCases','Fatalities'].sum().reset_index()

print(grouped.head(5))
import calendar

import numpy as np



grouped['Date'] = pd.to_datetime(grouped['Date'])

grouped['Month-Day'] = grouped['Date'].dt.strftime('%b-%d')

print(grouped.head(5))

ax = grouped.plot(x='Month-Day',y=['ConfirmedCases','Fatalities'], title= 'Actual confirmed cases and death of COVID-19',logy=False,logx=False,)

ax.set_ylabel('Total number of cases')

ax2 = grouped.plot(x='Month-Day',y=['ConfirmedCases','Fatalities'], title= 'Log Confirmed cases and death of COVID-19',logy=True)

ax2.set_ylabel('Total number of cases')

import geopandas as gpd

shapefile = '/kaggle/input/naturalearth/ne_50m_admin_0_countries.shp'

#Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

#Rename columns.

gdf.columns = ['country', 'country_code', 'geometry']

gdf.head()
import json

train_df.rename(columns={"Country_Region":"country"},inplace=True)

train_df.head(2)
grouped2 = train_df.groupby('country').sum().reset_index()



grouped2.head(5)

merged = gdf.merge(grouped2,on="country",how='left')

merged.head(5)

#print(merged[merged['country_code']=='USA'])

#print('USA' in merged['country_code'].tolist())
merged_json = json.loads(merged.to_json())

json_data = json.dumps(merged_json)
from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar

from bokeh.palettes import brewer

geosource = GeoJSONDataSource(geojson = json_data)

#Credit: https://towardsdatascience.com/a-complete-guide-to-an-interactive-geographical-map-using-python-f4c5197e23e0

#Define a sequential multi-hue color palette.

palette = brewer['YlGnBu'][8]

#Reverse color order so that dark blue is highest obesity.

palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

color_mapper = LinearColorMapper(palette = palette, low = merged['ConfirmedCases'].min(), high = merged['ConfirmedCases'].max())

#Define custom tick labels for color bar.

tick_labels = {'0': '0%', '5': '5%', '10':'10%', '15':'15%', '20':'20%', '25':'25%', '30':'30%','35':'35%', '40': '>40%'}

#Create color bar. 

color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,

border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

#Create figure object.

p = figure(title = 'Covid-19 cases', plot_height = 600 , plot_width = 950, toolbar_location = None)

p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None

#Add patch renderer to figure. 

p.patches('xs','ys', source = geosource,fill_color = {'field' :'ConfirmedCases', 'transform' : color_mapper},

          line_color = 'black', line_width = 0.25, fill_alpha = 1)

#Specify figure layout.

p.add_layout(color_bar, 'below')

#Display figure inline in Jupyter Notebook.

output_notebook()

#Display figure.

show(p)