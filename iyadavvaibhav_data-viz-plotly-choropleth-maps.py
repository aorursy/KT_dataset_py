import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import pandas as pd
data = dict(
    type = 'choropleth',
    locations = ['AZ','CA','NY'],
    locationmode = 'USA-states',
    colorscale = 'Portland',
    text = ['text1', 'text2', 'text3'],
    z = [1.0,2.0,3.0],
    colorbar = {'title':'Colorbar Title Here'}
)
data
layout = dict(
    geo = {'scope':'usa'}
)
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
