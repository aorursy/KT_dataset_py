# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
%matplotlib inline
pd.pandas.set_option('display.max_columns', None)
data= pd.read_csv('../input/choroplethplot/2014_World_Power_Consumption')
data.head(5)
dataB = dict(type = 'choropleth',
            colorscale = 'Viridis',
            reversescale = True,
            locations =data['Country'],
            locationmode = "country names",
            text=data['Text'],
            z=data['Power Consumption KWH'],
            colorbar = {'title':'Power Consumption KWH'})

layout = dict(
    title = 'Power Consumption for Countries',
    geo = dict(
        showframe = False,
        projection = {'type':'Mercator'}
    ))
choromap = go.Figure(data = [dataB],layout = layout)
iplot(choromap)
df=pd.read_csv('../input/election/2012_Election_Data')
df.head(5)
data = dict(type='choropleth',
            colorscale = 'YIOrRd',
            locations = df['State Abv'],
            z = df['Voting-Age Population (VAP)'],
            locationmode = 'USA-states',
            text = df['Voting-Age Population (VAP)'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Voting-Age Population (VAP)"}
            ) 
layout = dict(title = 'Voting-Age Population (VAP) by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
