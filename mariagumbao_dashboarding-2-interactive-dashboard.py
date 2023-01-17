# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as po
import plotly.graph_objs as go

# Load data
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')

# Data preparation
data = data.dropna() # Drop missing vlaues
data = data[data.business_latitude > 37] # Remove outliers
data.inspection_date = pd.to_datetime(data.inspection_date, format='%Y-%m-%dT%H:%M:%S') # Adjust date format
data.head()

# Group by restaurant
score_partial = data.groupby('business_id')['business_latitude','business_longitude',
                                    'inspection_score'].mean()
# Add names
names = data.business_name
names.index = data.business_id

score=score_partial.join(names.drop_duplicates(),how='left')

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Scatter plot in a map
graph_data = [
    go.Scattermapbox(
        lat=score.business_latitude,
        lon=score.business_longitude,
        mode='markers',
        marker=dict(
            size=7,
            color=score.inspection_score,
            colorscale='Viridis',
            colorbar=dict(
                title='Inspection Score'
            )
        ),
        text=score.business_name,
    )
]
# specify the layout of our figure
access_token='pk.eyJ1IjoibWFyaWFndW1iYW8iLCJhIjoiY2pwdmpwdGNyMGJzMTQzcWsyZjJjNHRzeiJ9.Xyr7tHeDm3VXHLMfYvPNHQ'
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title = "Restaurant Location",
    xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
    yaxis=dict(title="Inspection Score"),
    mapbox=dict(
        accesstoken=access_token,
        bearing=0,
        center=dict(
            lat=score.business_latitude.mean(),
            lon=score.business_longitude.mean()
        ),
        pitch=0,
        zoom=11,
        style='light'
    ),
)

# create and show our figure
fig = dict(data = graph_data, 
           layout = layout)
iplot(fig)
# Score relation with date of the inspection, is there bias?
date = data.groupby('business_id')['inspection_date','inspection_score'].max()
weekday = date.inspection_date.dt.weekday

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
graph_data = [go.Scatter(x=date.inspection_date, 
                         y=date.inspection_score,
                         mode='markers',
                         marker=dict(
                             color=weekday,
                             colorscale='Viridis',
                             showscale=True,
                             colorbar=dict(
                                 title='Weekday'
                             )
                         )
                        )
             ]

# specify the layout of our figure
layout = dict(title = "Score vs Time",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),
              yaxis=dict(title="Inspection Score")
             )

# create and show our figure
fig = dict(data = graph_data, 
           layout = layout)
iplot(fig)