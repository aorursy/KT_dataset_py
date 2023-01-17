# datetime operations

from datetime import timedelta



# for numerical analyis

import numpy as np



# to store and process data in dataframe

import pandas as pd



# basic visualization package

import matplotlib.pyplot as plt



# advanced ploting

import seaborn as sns



# interactive visualization

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots



# for offline ploting

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# color pallette

# Hexademical code RRGGBB (True Black #000000, True White #ffffff)

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
# list files

!ls ../input/corona-virus-report
# Country wise

country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')



# Replace missing values '' with NAN and then 0

country_wise = country_wise.replace('', np.nan).fillna(0)



country_wise.info()

country_wise.head(10)
# Grouped by day, country

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

full_grouped.info()

full_grouped.head(10)



# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

full_grouped.info()

# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('United Arab Emirates')



# Apply this mask to our original DataFrame to filter the required values.

uae = full_grouped[selected]

uae["New active"] = uae["Active"].diff()



uae.info()

uae.tail(10)
temp = uae.melt(id_vars="Date", value_vars=['New cases', 'New deaths'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()