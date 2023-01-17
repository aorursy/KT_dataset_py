import pandas as pd

import plotly.express as px

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#reading the File

data = pd.read_csv("/kaggle/input/singapore-airbnb/listings.csv")
#This shows top 5 rows form the table

data.head(5)
data.info()
data.describe()
#This shows the number of null values in every column

data.isnull().sum()
df1 = data.drop(columns="last_review")
df2 = data.drop(columns="reviews_per_month")
df2.isnull().sum()
df2



fig = px.scatter_mapbox(df2, lat="latitude", lon="longitude", hover_name="name", hover_data=["host_name", "room_type", "minimum_nights"], zoom=10, height=600,color="room_type", size="minimum_nights",

                  color_continuous_scale=px.colors.cyclical, size_max=15)

#color_discrete_sequence=["green"]

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()