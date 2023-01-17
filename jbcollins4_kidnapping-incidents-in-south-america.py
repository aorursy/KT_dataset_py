import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
# Read the entire global terrorism dataset.

df = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1')

print('%6d incidents in the entire dataset' % len(df))



# Get the rows that correspond to kidnappings in South America.

is_south_america = (df.region_txt) == 'South America'

is_kidnapping = (df.attacktype1 == 6) | (df.attacktype2 == 6) | (df.attacktype3 == 6)

sa = df[is_south_america & is_kidnapping]

sa = sa.dropna(axis=0, subset=['longitude', 'latitude'])

sa = sa.sort_values('city')

print('%6d kidnappings in South America' % len(sa))
# Here we will plot the locations of all kidnappings, with the dots colored to indicate

# the year in which they took place.

nn = len(sa)

year = sa.iyear

lon = sa.longitude + np.random.randn(nn) * 0.2

lat = sa.latitude + np.random.randn(nn) * 0.2

txt = sa.city + ', ' + year.astype(str)



cs = [[0.00, "rgb( 20,  20, 200)"], 

      [0.25, "rgb( 20, 200, 200)"],

      [0.50, "rgb( 20, 200,  20)"],

      [0.75, "rgb(200, 200,  20)"],

      [1.00, "rgb(200,  20,  20)"]]



dataset = dict(

           type = 'scattergeo',

           lon = list(lon),

           lat = list(lat),

           text = list(txt),

           mode = 'markers',

           name = '',

           hoverinfo = 'text',

           marker = dict(

               size = 8,

               opacity = 0.8,

               autocolorscale = False,

               symbol = 'square',

               line = dict( width=1, color="rgb(100, 100, 100)"),

               colorscale = cs,

               color = year,

               cmin = np.min(year),

               cmax = np.max(year),

               colorbar = dict(title = "year of incident")

           ))

        

layout = dict(

         title = 'Kidnappings in South America (1970-2015)',

         autosize = False,

         width = 800,

         height = 1000,

         showlegend = False,

         geo = dict(

             scope = 'south america',

             projection = dict(type='Mercator'),

             showland = True,

             landcolor = 'rgb(250, 250, 250)',

             subunitwidth = 1,

             subunitcolor = 'rgb(217, 217, 217)',

             countrywidth = 1,

             countrycolor = 'rgb(217, 217, 217)',

             showcountries = True

         ))



figure = dict(data = [dataset], layout = layout)

iplot(figure)