# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/alldatasets/AB_NYC_2019.csv')
df.head()
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="price", size='price',

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

fig.show()
df.shape
df.dtypes
df.isnull().sum().sort_values(ascending=False)
df['price'].value_counts()
df['room_type'].value_counts()
df['name'].value_counts()
df['host_id'].value_counts()
df['neighbourhood'].value_counts()
df['neighbourhood_group'].value_counts()
df["reviews_per_month"].value_counts()
fig = px.density_mapbox(df, lat='latitude', lon='longitude',z='price' ,radius=10, zoom=10,

                        mapbox_style="stamen-terrain")

fig.show()
fig = px.histogram(df, x="number_of_reviews")

fig.show()
fig = px.histogram(df, x="price")

fig.show()
fig = px.histogram(df, x="neighbourhood_group")

fig.show()
fig = px.histogram(df, x="neighbourhood", color="neighbourhood_group")

fig.show()
fig = px.box(df, x="neighbourhood_group", y="price", color="room_type")

fig.show()
fig = px.scatter(df, x="number_of_reviews", y="price", color="neighbourhood_group")

fig.show()
fig = px.histogram(df, x="price", color="neighbourhood_group")

fig.show()