# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
zomato_data = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")

zomato_data.head()
countryCode = pd.read_excel('../input/Country-Code.xlsx')

countryCodeCountry = pd.merge(zomato_data, countryCode, on = 'Country Code')

countryCodeCountry['Country'].value_counts().head(5)
countryCodeCountry['City'].value_counts().head()
countryCodeCountry.groupby('Cuisines')['Aggregate rating'].mean().head(100).plot(kind='bar', figsize = (20, 7))
countryCodeCountry.groupby('Price range')['Aggregate rating'].mean().plot(kind = 'bar', figsize = (20, 7))
print('No:')

print(zomato_data[zomato_data['Has Table booking'] == 'No']['Aggregate rating'].mean())

print('Yes:')

print(zomato_data[zomato_data['Has Table booking'] == 'Yes']['Aggregate rating'].mean())
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.graph_objs as go





plot_data = [dict(

    type='scattergeo',

    lon = zomato_data['Longitude'],

    lat = zomato_data['Latitude'],

    text = zomato_data['Restaurant Name'],

    mode = 'markers',

    marker = dict(

        cmin = 0,

        color = zomato_data['Aggregate rating'],

        cmax = zomato_data['Aggregate rating'].max(),

        colorbar=dict(

                    title="Rating"

                )

    )

    

)]

fig = go.Figure(data=plot_data)

iplot(fig)
data = ['Price range', 'Votes', 'Aggregate rating']

a = zomato_data[data].corr(method='spearman')

sns.heatmap(a)