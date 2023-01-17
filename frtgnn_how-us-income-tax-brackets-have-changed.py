import os

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)



warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/historical-income-tax-rates-brackets/tax_over_years.csv",sep=";")

df.drop(['Unnamed: 5','Unnamed: 6'],axis=1,inplace=True)

df['Bottom Bracket Rate %'] = df['Bottom Bracket Rate %'].astype(float)

df['Top Bracket Rate %']    = df['Top Bracket Rate %'].astype(float)

df.head(31).style.background_gradient(cmap='Reds')
df_simple = pd.DataFrame()



df_simple['Year'] = df['Year']

df_simple['Low Income Max Tax']  = df['Bottom Bracket Rate %'] * df['Bottom Bracket Taxable Income up to'] / 100

df_simple['High Income Max Tax'] = df['Top Bracket Rate %'] * df['Top Bracket Taxable Income Over'] / 100

#df_simple.set_index('Year',inplace=True)

df_simple.head(31).style.background_gradient(cmap='Greens')
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x = df['Year'].head(50),y = df['Bottom Bracket Rate %'].head(50),

                    mode='lines+markers',

                    name='Bottom Bracket Rate %'))

fig.add_trace(go.Scatter(x = df['Year'].head(50),y = df['Top Bracket Rate %'].head(50),

                    mode='lines+markers',

                    name='Top Bracket Rate %'))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)



fig.update_layout(autosize=False, width=800,height=500, legend_orientation="h", template = 'plotly_dark')



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x = df['Year'].head(50),y = df['Bottom Bracket Taxable Income up to'].head(50),

                    mode='lines+markers',

                    name='Bottom Bracket Taxable Income up to'))

fig.add_trace(go.Scatter(x = df['Year'].head(50),y = df['Top Bracket Taxable Income Over'].head(50),

                    mode='lines+markers',

                    name='Top Bracket Income Over'))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)



fig.update_layout(autosize=False, width=800,height=500, legend_orientation="h", template = 'plotly_dark')



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x = df_simple['Year'].head(50),y = df_simple['Low Income Max Tax'].head(50),

                    mode='lines+markers',

                    name='Low Income Max Tax'))

fig.add_trace(go.Scatter(x = df_simple['Year'].head(50),y = df_simple['High Income Max Tax'].head(50),

                    mode='lines+markers',

                    name='High Income Max Tax'))

fig.update_traces(mode='lines+markers', marker_line_width=2, marker_size=10)



fig.update_layout(autosize=False, width=800,height=500, legend_orientation="h", template = 'plotly_dark')



fig.show()