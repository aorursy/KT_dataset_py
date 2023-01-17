import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/indias-gdp-statewise/Indias GDP (Statewise).csv')
df.head()
fig = px.treemap(df, 
                 path=['State_UT'],
                 values = 'Nominal GDP(Trillion INR)',
                 hover_data = ['Nominal GDP(Billion USD)'],
                 names='State_UT', 
                 color='Nominal GDP(Trillion INR)',
                 height=600,
                 color_continuous_scale='RdBu',
                 title='Indias GDP - Statewise',
                )

fig.show()
fig = px.pie(df, 
             values='Nominal GDP(Trillion INR)', 
             names='State_UT')
fig.show()
fig = px.scatter(df, 
                 x="Rank", y='Rank',
	             size="Nominal GDP(Trillion INR)",
                 hover_name="State_UT", 
                 log_x=True, 
                 color="State_UT",
                 size_max=60)
fig.show()
