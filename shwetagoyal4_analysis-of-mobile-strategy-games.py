# Import necessary packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objects as go

import plotly.express as px



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.offline as py

py.init_notebook_mode(connected = True)



import warnings

warnings.filterwarnings('ignore')



from wordcloud import WordCloud
data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')             
data.head()
# Checking missing/null values



data.isna().sum()
fig = go.Figure([go.Bar(x=data["Primary Genre"], y=data["Average User Rating"])])

fig.update_layout(title_text="Distribution of genres")

py.iplot(fig, filename="test") 
fig = px.scatter(data, x = "Age Rating", y = "User Rating Count", title="Distribution of rating")

py.iplot(fig, filename="test")
fig = px.scatter(data, y = "Size", x = "Price")

py.iplot(fig, filename="test")
fig = px.scatter(data, y="In-app Purchases", x="Size")

py.iplot(fig, filename="test")