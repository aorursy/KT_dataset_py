import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_json('../input/train.json')
df.loc[df['bathrooms'] <= 0,'bathrooms'] = 1

df.loc[df['bedrooms'] <= 0,'bedrooms'] = 1



df.loc[df['bathrooms'] > 3,'bathrooms'] = 3

df.loc[df['bedrooms'] > 3,'bedrooms'] = 3



df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']



df = df[df['price'] < 10000]



trans = {

    'low':'red',

    'medium': 'green',

    'high':'blue'

}

colors = [trans[x.interest_level] for i,x in df.iterrows()]



plt.scatter(df.price, df.bedrooms_bathrooms, c=colors)



#plt.yticks(np.arange(0,40,1))

plt.yticks([11,12,13,14,15,21,22,23,24,31,32,33,34])

plt.show()

#This new transformation allows combine the bedrooms and bathrooms. 

#For example 32(y axis) means 3 bedrooms with 2 bathrooms. 

#In fact this new feature improve the score in the LB, a tiny improvment.
high_df = df[df["interest_level"] == "high"]

plt.scatter(high_df.price, high_df.bedrooms_bathrooms, c="blue")



#plt.yticks(np.arange(0,40,1))

plt.yticks([11,12,13,14,15,21,22,23,24,31,32,33,34])

plt.show()
medium_df = df[df["interest_level"] == "medium"]

plt.scatter(medium_df.price, medium_df.bedrooms_bathrooms, c="green")



#plt.yticks(np.arange(0,40,1))

plt.yticks([11,12,13,14,15,21,22,23,24,31,32,33,34])

plt.show()
from bokeh.plotting import figure,show

from bokeh.io import output_file, push_notebook, output_notebook

from bokeh.models import (

    GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool

)

output_notebook()
map_options = GMapOptions(lat=40.75, lng=-74.00, map_type="roadmap", zoom=11)



plot = GMapPlot(

    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options

)

plot.title.text = "New York"



plot.api_key = "put your key" #YOUR KEY GOES HERE



#high

sourceHigh = ColumnDataSource(

    data=dict(

        lat=df[df['interest_level'] == 'high'].latitude,

        lon=df[df['interest_level'] == 'high'].longitude,

    )

)

circleHigh = Circle(x="lon", y="lat", size=2, fill_color="#000099", fill_alpha=0.8, line_color=None)

# medium

sourceMedium = ColumnDataSource(

    data=dict(

        lat=df[df['interest_level'] == 'medium'].latitude,

        lon=df[df['interest_level'] == 'medium'].longitude,

    )

)

circleMedium = Circle(x="lon", y="lat", size=2, fill_color="yellow", fill_alpha=0.8, line_color=None)

# low

sourceLow = ColumnDataSource(

    data=dict(

        lat=df[df['interest_level'] == 'low'].latitude,

        lon=df[df['interest_level'] == 'low'].longitude,

    )

)

circleLow = Circle(x="lon", y="lat", size=2, fill_color="red", fill_alpha=0.8, line_color=None)



plot.add_glyph(sourceHigh, circleHigh)

#plot.add_glyph(sourceMedium, circleMedium)

#plot.add_glyph(sourceLow, circleLow)



plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

show(plot,  notebook_handle=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print(df[df['interest_level']=='high'].price.describe())

print(df[df['interest_level']=='medium'].price.describe())

print(df[df['interest_level']=='low'].price.describe())