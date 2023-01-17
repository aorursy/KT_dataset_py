# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# color pallette

cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue
mers_cntry = pd.read_csv("../input/mers-outbreak-dataset-20122019/country_count_latest.csv")

mers_weekly = pd.read_csv("../input/mers-outbreak-dataset-20122019/weekly_clean.csv")
mers_cntry.head()
mers_weekly.head()
mers_weekly['Year-Week'] = mers_weekly['Year'].astype(str) + ' - ' + mers_weekly['Week'].astype(str)

mers_weekly['Date'] = pd.to_datetime(mers_weekly['Week'].astype(str) + 

                                     mers_weekly['Year'].astype(str).add('-1'),format='%V%G-%u')

mers_weekly.head()
mers_weekly.isna().sum()
fig = px.choropleth(mers_cntry, locations="Country", locationmode='country names',

                    color="Confirmed", hover_name="Country", title='Confirmed Cases', 

                    color_continuous_scale="Sunset")

fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.treemap(mers_cntry, path=["Country"], values="Confirmed", title='Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Dark2)

fig.show()
fig = px.bar(mers_weekly, x="Date", y="New Cases", color='Region',

             title='Number of new cases every week')

fig.update_xaxes(tickangle=-90)

fig.show()