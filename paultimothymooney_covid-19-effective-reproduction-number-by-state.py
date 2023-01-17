# Import Python Packages

import pandas as pd

import numpy as np

import plotly.express as px

# Load Data

df = pd.read_csv('/kaggle/input/effective-reproduction-number-rt-by-us-state/rt.csv')

df_today = df[df.date==np.max(df.date)]

# Plot Data

fig = px.choropleth(df_today, 

                    locations="region", 

                    color="median", 

                    locationmode = 'USA-states', 

                    hover_name="region",

                    range_color=[0.8,1.2],scope="usa",

                    title='COVID-19 effective reproduction number by state')

fig.show()