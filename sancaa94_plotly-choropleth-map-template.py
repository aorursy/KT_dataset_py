import pandas as pd

import plotly.plotly as py

import plotly.graph_objs as go #go.figure

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
#z is floats, these valuse would be the actaul data numbers

vals = list(range(1,51))

z = []

for i in vals:

    i = float(i)

    z.append(i)

print(z)
# 1. build a data dictionary cast a list into a dictionary

data = dict(

    type = 'choropleth', #key type

    locations = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],

    locationmode = 'USA-states', #lets plotly know its USA

    colorscale = 'Jet',

    text = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 

          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 

          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 

          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 

          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], #a list of what hovers over each of the locations

    z = z, # equal to the values that are going to be shown to you in an actual color scale

    colorbar = {'title':'Colorbar Title Here'}

)

#text must be in the same index location as locations
data
layout = dict(geo = {'scope':'usa'})
choromap = go.Figure(data = [data],layout=layout)
iplot(choromap)