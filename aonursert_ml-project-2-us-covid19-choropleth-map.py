!pip install chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
%matplotlib inline

import pandas as pd

init_notebook_mode(connected=True) 
import os
print(os.listdir("../input"))
# Read US Covid19 Total Cases by State
df = pd.read_csv("../input/may-2020-us-covid19-total-cases-by-state/us_covid19.csv")
df.head()
# Create data using dictionary for the Choropleth arguments
data = dict(type = "choropleth",
            colorscale = "YlOrRd",
            locations = df["Codes"],
            z = df["Total Cases"],
            locationmode = "USA-states",
            text = df["States"],
            marker = dict(line = dict(color = "rgb(255,255,255)", width = 2)),
            colorbar = {"title":"Total Cases"}
            ) 
# Make layout using dictionary with the arguments
layout = dict(title = "May 2020 US Covid19 Total Cases by State",
              geo = dict(scope="usa", showlakes = True, lakecolor = "rgb(85,173,240)")
             )
# Create choromap using data and layout
choromap = go.Figure(data = [data], layout = layout)
# Displaying the Result Choropleth Map using iplot
iplot(choromap)