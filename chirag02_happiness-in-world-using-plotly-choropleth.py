import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
init_notebook_mode(connected=True)
df_2015 = pd.read_csv('../input/world-happiness/2015.csv')

df_2015.head()
df_2015.info()
df_2016 = pd.read_csv('../input/world-happiness/2016.csv')

df_2016.head()
df_2016.info()
df_2017 = pd.read_csv('../input/world-happiness/2017.csv')

df_2017.head()
df_2017.info()
df_2015 = df_2015.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])

df_2016 = df_2016.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])

df_2017 = df_2017.replace(['United States', 'Russia'], ['United States of America', 'Russian Federation'])
df_code = pd.read_csv('../input/country-code/country_code.csv')

df_code.head()
df_code = df_code.drop(['Unnamed: 0','code_2digit'], axis=1)

df_code.head()
df_code = df_code.rename(index=str, columns={'Country_name':'Country'})
df_2015 = df_2015.merge(df_code, how='inner', on='Country')

df_2016 = df_2016.merge(df_code, how='inner', on='Country')

df_2017 = df_2017.merge(df_code, how='inner', on='Country')
df_2015.info()
def plotmap(df, plot_series, color_title, scl, title ):

    data = dict(type = 'choropleth',

               locations = df['code_3digit'],

               z = df[plot_series],

               text = df['Country'],

               colorbar = {'title':color_title},

               colorscale = scl)



    layout = dict(title=title,

                geo=dict(showframe=False, projection={'type':'equirectangular'}))



    fig = go.Figure(data = [data], layout = layout)

    iplot(fig)
plotmap(df_2015, 'Happiness Rank', 'Happiness Rank 2015', 'RdBu', '2015 Happiness Ranking')
plotmap(df_2016, 'Happiness Rank', 'Happiness Rank 2016', 'RdBu', '2016 Happiness Ranking')
plotmap(df_2017, 'Happiness.Rank', 'Happiness Rank 2017', 'RdBu', '2017 Happiness Ranking')
plotmap(df_2015, 'Economy (GDP per Capita)', 'GDP per capita', 'Greens', '2015 GDP per Capita')
plotmap(df_2016, 'Economy (GDP per Capita)', 'GDP per capita', 'Greens', '2016 GDP per Capita')
plotmap(df_2017, 'Economy..GDP.per.Capita.', 'GDP per capita', 'Greens', '2017 GDP per Capita')
plotmap(df_2015, 'Family', 'Family score', 'Blues', '2015 Family Score')
plotmap(df_2016, 'Family', 'Family score', 'Blues', '2016 Family Score')
plotmap(df_2017, 'Family', 'Family score', 'Blues', '2017 Family Score')
plotmap(df_2015, 'Health (Life Expectancy)', 'Life Expectancy score', 'YlGnBu', '2015 Life Expectancy Score')
plotmap(df_2016, 'Health (Life Expectancy)', 'Life Expectancy score', 'YlGnBu', '2016 Life Expectancy Score')
plotmap(df_2017, 'Health..Life.Expectancy.', 'Life Expectancy score', 'YlGnBu', '2017 Life Expectancy Score')
plotmap(df_2015, 'Freedom', 'Freedom score', 'Greys', '2015 Freedom Score')
plotmap(df_2016, 'Freedom', 'Freedom score', 'Greys', '2016 Freedom Score')
plotmap(df_2017, 'Freedom', 'Freedom score', 'Greys', '2017 Freedom Score')
plotmap(df_2015, 'Trust (Government Corruption)', 'Government Corruption score', 'Hot', '2015 Government Corruption Score')
plotmap(df_2016, 'Trust (Government Corruption)', 'Government Corruption score', 'Hot', '2016 Government Corruption Score')
plotmap(df_2017, 'Trust..Government.Corruption.', 'Government Corruption score', 'Hot', '2017 Government Corruption Score')
plotmap(df_2015, 'Generosity', 'Generocity score', 'Reds', '2015 Generocity Score')
plotmap(df_2016, 'Generosity', 'Generocity score', 'Reds', '2016 Generocity Score')
plotmap(df_2017, 'Generosity', 'Generocity score', 'Reds', '2017 Generocity Score')