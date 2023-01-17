# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import unicodedata

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/nyc-precovid-restaurant-data/Pre_restaurants.csv")

df.head()
g = sns.lmplot(x="LONGITUDE", y="LATITUDE", data=df,

           fit_reg=False, scatter_kws={"s": 30}, hue='Borough', height=10)

plt.title('Pre-Covid Restaurants NY')

plt.show()
plt.figure(figsize=(20,12))

g = sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=df, hue='income level')

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
fig_px = px.scatter_mapbox(df, lat="LATITUDE", lon="LONGITUDE",

                           hover_name="Borough",

                           zoom=11, height=300)

fig_px.update_layout(mapbox_style="open-street-map",

                     margin={"r":0,"t":0,"l":0,"b":0})



fig_px.show()
fig_px.update_traces(marker={"size": [10 for x in df]})
fig = px.bar(df, x= "GRADE DATE", y= "New York", title= "Pre-Covid Restaurants NY", color_discrete_sequence=['crimson'],)

fig.show()