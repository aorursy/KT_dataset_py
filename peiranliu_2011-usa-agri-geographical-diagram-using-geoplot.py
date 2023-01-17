# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



# Any results you write to the current directory are saved as output.
data = dict(type = 'choropleth',

            locations = ['AZ','CA','NY'],

            locationmode = 'USA-states',

            colorscale= 'Portland',

            text= ['text1','text2','text3'],

            z=[1.0,2.0,3.0],

            colorbar = {'title':'Colorbar Title'})
layout = dict(geo = {'scope':'usa'})
#set up choromap

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
df = pd.read_csv('../input/2011usagri/2011_US_AGRI_Exports')

df.head()

#reading data head() and import data
data = dict(type='choropleth',

            colorscale = 'YLOrRd',

            locations = df['code'],

            z = df['total exports'],

            locationmode = 'USA-states',

            text = df['text'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"Millions USD"}

            ) 
layout1= dict(title = '2011 US Agriculture Exports by State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap = go.Figure(data = [data],layout = layout1)
iplot(choromap)
df = pd.read_csv('../input/2014-world-gdp/2014_World_GDP')

df.head()
data = dict(

        type = 'choropleth',

        locations = df['CODE'],

        z = df['GDP (BILLIONS)'],

        text = df['COUNTRY'],

        colorbar = {'title' : 'GDP Billions US'},

      ) 
layout = dict(

    title = '2014 Global GDP',

    geo = dict(

        showframe = False,

        projection = {'type':'mercator'}

    )

)
choromap3 = go.Figure(data = [data],layout = layout)

iplot(choromap3)