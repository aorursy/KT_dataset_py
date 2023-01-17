# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import os

from bokeh.io import show, output_file

from bokeh.models import ColumnDataSource, Legend, LegendItem, Scatter

from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.models.tools import HoverTool

from bokeh.core.properties import value

from bokeh.palettes import Spectral10, Category20, Category20_17, inferno, magma, viridis

import matplotlib.pyplot as plt

from bokeh.transform import jitter



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dtype = {'DayOfWeek': np.uint8, 'DayofMonth': np.uint8, 'Month': np.uint8 , 

         'Cancelled': np.uint8, 'Year': np.uint16, 'FlightNum': np.uint16 , 

         'Distance': np.uint16, 'UniqueCarrier': str, 'CancellationCode': str, 

         'Origin': str, 'Dest': str, 'ArrDelay': np.float16, 

         'DepDelay': np.float16, 'CarrierDelay': np.float16, 

         'WeatherDelay': np.float16, 'NASDelay': np.float16, 

         'SecurityDelay': np.float16, 'LateAircraftDelay': np.float16, 

         'DepTime': np.float16}
path = '../input/btstats/2008.csv'

flights_df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype)
flights_df.head()
flights_df = flights_df[np.isfinite(flights_df['DepTime'])]

flights_df['DepHour'] = flights_df['DepTime'] // 100

flights_df['DepHour'].replace(to_replace=24, value=0, inplace=True)

flights_df['DepMin'] = flights_df['DepTime'] - flights_df['DepHour']*100
flights_df['DepHour'] = flights_df['DepHour'].apply(lambda f: format(f, '.0f'))

flights_df['DepMin'] = flights_df['DepMin'].apply(lambda f: format(f, '.0f'))
flights_df['Date'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])



flights_df['DateTime'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day', 'DepHour': 'Hour', 'DepMin':'Minute'})\

                                        [['Year', 'Month', 'Day', 'Hour', 'Minute']])
num_flights_by_date = flights_df.groupby('Date').size().reset_index()

num_flights_by_date.columns = ['Date', 'Count']
TOOLS = "pan, wheel_zoom, box_zoom, box_select,reset, save" # the tools you want to add to your graph

source = ColumnDataSource(num_flights_by_date) # data for the graph
# Graph has date on the x-axis

p = figure(title="Graph 1: Number of flights per day in 2008", x_axis_type='datetime',tools = TOOLS)



p.line(x='Date', y='Count', source=source) #build a line chart

p.xaxis.axis_label = 'Date'

p.yaxis.axis_label = 'Number of flights'



p.xgrid.grid_line_color = None



# add a hover tool and show the date in date time format

hover = HoverTool()

hover.tooltips=[

    ('Date', '@Date{%F}'),

    ('Count', '@Count')

]

hover.formatters = {'Date': 'datetime'}

p.add_tools(hover)

output_notebook() # show the output in jupyter notebook

show(p)
# for the sake of image clarity, let us take only a couple of months data

df = flights_df[flights_df['Date']<'03-01-2008']

ct = pd.crosstab(df.Date, df.UniqueCarrier)

carriers = ct.columns.values #list of the carriers
ct = ct.reset_index() # we want to make the date a column

ct['Date'] = ct['Date'].astype(str) # to show it in the x-axis
# Graph has date on the x-axis 



source = ColumnDataSource(data=ct) # data for the graph

Date = source.data['Date']



#legend = Legend(items=[LegendItem(legend_data)], position=(0,-30))



# x_range : specifies the x-axis values, in our case Date

p = figure(x_range=Date, title="Graph 2: Flights in the first 2 months of 2008, by carrier",\

           tools = TOOLS, width=750)



renderers = p.vbar_stack(carriers, x='Date', source=source, width=0.5, color=magma(20), \

             legend=[value(x) for x in carriers])



p.xaxis.axis_label = 'Date'

p.yaxis.axis_label = 'Number of flights'



p.xgrid.grid_line_color = None



p.y_range.start = 0

p.y_range.end = 25000 #to make room for the legend

p.x_range.range_padding = 0.1

p.xgrid.grid_line_color = None

p.axis.minor_tick_line_color = None

p.outline_line_color = None



#add hover

hover = HoverTool()

hover.tooltips=[

    ('Date', '@Date{%F}'),

    ('Carrier', '$name'), #$name provides data from legend

    ('Count', '@$name') #@$name gives the value corresponding to the legend

]

hover.formatters = {'Date': 'datetime'}

p.add_tools(hover)

p.xaxis.major_label_orientation = math.pi/2 #so that the labels do not come on top of each other



#move legend outside the plot so that it does not interfere with the data

# creating external legend did not work

# so doing a roundabout of creating an intenal legend, copying it over to a new legend

# placing it on right and nulling te internal legend

new_legend = p.legend[0]

p.legend[0].plot = None

p.add_layout(new_legend, 'right')

p.legend.click_policy="hide"



output_notebook()

show(p)
# Let us next plot a bubble chart of the flights between 2 cities.



df = flights_df[['Origin', 'Dest', 'UniqueCarrier', 'Distance']]

df['Flight'] = df['Origin']+'-'+df['Dest'] # new variable for flight
df.head()
#find number of flights  between any 2 cities and sort them

df_by_flight = df.groupby(['Flight']).agg({'Flight': 'count'}).sort_values(('Flight'), ascending=False)
df_by_flight.head()
df_by_flight.columns=['Count']

df_by_flight = df_by_flight.reset_index()



#merge it back with df to get other columns of interest

df_new = df_by_flight.merge(df, on='Flight')
df_new = df_new.drop_duplicates(subset=['Flight'])
df_new.head()
df_new.describe()
df_new = df_new.drop(columns=['Flight', 'UniqueCarrier', 'Distance'])
df_new.head()
df_new = df_new[0:100]
df_new['Count_gr'] = df_new['Count']/5000 #for the sake of charting
source = ColumnDataSource(data=df_new)

Origin_l = df_new['Origin'].unique()

Dest_l = df_new['Dest'].unique()



p = figure(title='Graph 3: Flights between 2 cities (top 100 values only)',x_range=Origin_l, y_range=Dest_l, tools=TOOLS, width=750)



p.circle(x='Origin', y='Dest', radius='Count_gr',

          fill_color='purple', fill_alpha=0.4, source=source,

          line_color=None)



p.x_range.range_padding = 0.5

p.y_range.range_padding = 0.5



#add hover

hover = HoverTool()

hover.tooltips=[

    ('From', '@Origin'),

    ('To', '@Dest'),

    ('Count', '@Count') #@$name gives the value corresponding to the legend

]



p.add_tools(hover)

p.xaxis.major_label_orientation = math.pi/2



output_notebook()

show(p)