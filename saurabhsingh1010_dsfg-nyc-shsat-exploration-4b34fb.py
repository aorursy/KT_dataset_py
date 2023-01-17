import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import seaborn as sns

color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import folium

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
import os
print(os.listdir("../input"))
school_df = pd.read_csv("../input/2016 School Explorer.csv")
print("Number or rows & columns : ", school_df.shape)
school_df.head()
NYC_COORDINATES = (40.767937, -73.982155)

# create empty map zoomed in on NYC
map_1 = folium.Map(location=NYC_COORDINATES, zoom_start=10, tiles='Stamen Toner')
        
display(map_1)
trace1 = go.Scatter(
    y = school_df["Latitude"].values,
    x = school_df["Longitude"].values,
    mode='markers',
    marker=dict(
        size=10,
        color = school_df["District"].values, #set color equal to a variable
        colorscale='Jet',
        #showscale=True
    ),
    text = school_df["School Name"].values
)
layout = go.Layout(
    autosize=False,
    #plot_bgcolor='rgba(240,240,240,1)',
    plot_bgcolor='rgba(255,160,122,0.1)',
    width=800,
    height=800,
    title = "Location of schools - color coded by District code"
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter-plot-with-colorscale')
school_df["School Income Estimate"] = school_df["School Income Estimate"].apply(lambda x: float(str(x).replace("$","").replace(",","")))

trace1 = go.Scatter(
    y = school_df["Latitude"].values,
    x = school_df["Longitude"].values,
    mode='markers',
    marker=dict(
        size=10,
        color = school_df["School Income Estimate"].values, #set color equal to a variable
        colorscale='Greens',
        showscale=True,
        reversescale=True
    ),
    text = school_df["School Name"].values
)
layout = go.Layout(
    autosize=False,
    plot_bgcolor='rgba(255,160,122,0.1)',
    width=800,
    height=800,
    title = "Location of schools - color coded by Income"
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter-plot-with-colorscale')
