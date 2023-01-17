# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt                   # For graphics

from matplotlib import animation

import matplotlib.animation as animation

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
g = '/kaggle/input/gold-reserves-by-country-quarterly/gold_quarterly_reserves_ounces.csv'

gold = pd.read_csv(g, index_col='Country Name')
gold.info()
gold.head()
time = gold['Time Period'].str.split('Q', 1).str

gold['Year'] = time[0]

gold['Quarter'] = time[1]

gold = gold.drop('Time Period',axis=1)



gold.head()
gold['Value'] = gold['Value']/32000
gold['Year'] = gold['Year'].astype(int)

gold['Value'] = round(gold['Value'],0)
gold['Year']
gold2 = gold.drop(['Advanced Economies','Sub-Saharan Africa','Central African Economic and Monetary Community','CIS','Emerging and Developing Asia','Emerging and Developing Europe','Emerging and Developing Countries', 'Europe', 'Euro Area','Middle East, North Africa, Afghanistan, and Pakistan','World','Western Hemisphere','West African Economic and Monetary Union (WAEMU)'])
gold2.index.unique()
gold2 = gold2.loc[gold2['Quarter']=='4']
gold2['Country Name'] = gold2.index

gold2.reset_index(drop=True, inplace=True)
country = gold2['Country Name'].unique()
temp = {i: j for j, i in enumerate(set(gold2['Country Name']))} 

gold2['Color'] = [temp[i] for i in gold2['Country Name']] 
gold2.sort_values(['Year','Value'], ascending=[True,False], inplace=True)
gold2['Country Name'] = gold2['Country Name'].replace(['Venezuela, Republica Bolivariana de', 'China, P.R.: Mainland', 'Taiwan Province of China', 'Russian Federation', 'Iran, Islamic Republic of'],['Venezuela', 'China', 'Taiwan', 'Russia', 'Iran'])
gold3 = pd.concat(gold2[gold2['Year']==i][:20] for i in gold2['Year'].unique())
gold3.sort_values(['Year', 'Value'], ascending=True, inplace=True)
fig = px.bar(gold3, x='Value', y='Country Name', title='Gold', animation_frame='Year', orientation='h', text='Value', width=1100, height=800, color='Color', color_continuous_scale=px.colors.qualitative.Alphabet)



fig.update_layout(xaxis=dict(title='Tonnes of gold', showgrid=True, gridcolor='grey'), yaxis=dict(title=''), paper_bgcolor='white', plot_bgcolor='white', coloraxis=dict(showscale=False))

fig.show()
topgold2 = gold2.loc[gold2['Country Name'].isin(['United States','United Kingdom', 'Switzerland', 'Germany', 'France', 'Italy', 'Russian Federation', 'China, P.R.: Mainland', 'Taiwan province of China'])].sort_values('Year')
fig = px.line(topgold2, x='Year', y='Value', title='Top Gold Reserves', color='Country Name')



fig.update_layout(xaxis=dict(title='Year', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white', legend=dict(xanchor='right', yanchor='top'))

fig.show()
names2018 = gold2.loc[gold2['Year']==2018][:20]
names2018 = names2018['Country Name'].values
names2018
top2018 = gold2.loc[gold2['Country Name'].isin(names2018)]
top2018_x = top2018.sort_values(['Country Name', 'Year'], ascending=True)

top2018_x['Variation'] = top2018_x['Value'] - top2018_x['Value'].shift(1)

top2018_x = top2018_x[top2018_x['Country Name'].duplicated(keep='first')]
top2018_x['Variation']
fig = px.bar(top2018_x, x='Year', y='Variation', title='Top Gold Reserves variations', color='Country Name')



fig.update_layout(xaxis=dict(title='Year', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white', legend=dict(orientation='h'))

fig.show()
var_til1972 = top2018_x.loc[(top2018_x['Year']<1972)].groupby('Country Name').sum().sort_values('Variation', ascending=False)
var_til1972['Country Name'] = var_til1972.index

var_til1972.reset_index(drop=True, inplace=True)
fig = px.bar(var_til1972, x='Country Name', y='Variation', title='Total Gold reserves variation 1950-1972', color='Variation')



fig.update_layout(xaxis=dict(title='Country', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white')

fig.show()
diffplus = var_til1972.loc[var_til1972['Country Name'].isin(['Germany', 'France', 'Italy', 'Netherlands', 'Switzerland', 'Portugal', 'Austria', 'Japan', 'Spain', 'Lebanon', 'Saudi Arabia', 'Taiwan Province of China'])]

diffplus = diffplus.sum()
diffUKUS = var_til1972.loc[var_til1972['Country Name'].isin(['United Kingdom', 'United States'])]

diffUKUS = diffUKUS.sum()

diffUKUS = diffUKUS * (-1) #multiply by -1 to convert value to positive to have a better visualization in the plot
diffcomp = pd.DataFrame([diffplus.Variation, diffUKUS.Variation])
diffcomp = diffcomp.rename({0:'Variation'}, axis='columns')

diffcomp['Groups'] = ['Rest','UK US']
diffcomp
fig = px.bar(diffcomp, x='Groups', y='Variation', title='Comparation of increasing and decreasing countries from 1950 to 1972', color='Groups')



fig.update_layout(xaxis=dict(title='Group of countries', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white')

fig.show()