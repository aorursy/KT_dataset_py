import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

df_w_consumption = pd.read_csv('../input/water-consumption-in-the-new-york-city.csv', index_col='Year')

print(df_w_consumption.info())

print(df_w_consumption.head())
#https://github.com/santosjorge/cufflinks/issues/185

!pip install plotly

!pip install cufflinks

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()
#https://stackoverflow.com/questions/46016712/plotly-change-figure-size-by-calling-cufflinks-in-pandas

layout1 = cf.Layout(

    height=800,

    width=1000

)

_=df_w_consumption.iplot(logy=True)
df_w_consumption.iplot(kind='box', subplots=True)
df_w_consumption[['NYC Consumption(Million gallons per day)', 'Per Capita(Gallons per person per day)']].iplot(kind='hist', subplots=True, bins=7)
df_w_consumption[['NYC Consumption(Million gallons per day)', 'Per Capita(Gallons per person per day)']].iplot(kind='hist', subplots=True, bins=7, histnorm='probability')
df_w_consumption.iplot(subplots=True)