# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""import plotly.plotly as py

import plotly.graph_obj as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot"""

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)
data = dict(type='choropleth',

           locations = ['AZ','CA','NY'],

           locationmode = 'USA-states',

           colorscale = 'Portland',

           text = ['text 1','text 2','text 3'],

           z = [1.0,2.0,3.0],

           colorbar = {'title':'Colorbar Title Goes Here'})        # In place of Portland we can use Greens 
layout = dict(geo={'scope':'usa'})
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
df= pd.read_csv('../input/2011_US_AGRI_Exports')

df.head()
data1 = dict(type='choropleth',

           locations = df['code'],

           locationmode = 'USA-states',

           colorscale = 'Portland',

           text = df['text'],

           z = df['total exports'],

           colorbar = {'title':'Millions USD'})    
layout1 = dict(title = '2011 US Agriculture Exports by State',

             geo = dict(scope = 'usa',showlakes = True,lakecolor = 'rgb(85,173,240)'))
layout1
chromap2 = go.Figure(data = [data1],layout =layout1)
iplot(chromap2)
df2= pd.read_csv('../input/2014_World_GDP')

df2.head()
data = dict(type = 'choropleth',

           locations = df2['CODE'],

           z=df2['GDP (BILLIONS)'],

           text = df2['COUNTRY'],

           colorbar = {'title':'GDB in Billions USD'})



layout = dict(title='2014 Global GDP',

             geo = dict(showframe= False,

                       projection = {'type': 'mercator'}))
choromap3 = go.Figure(data=[data],layout=layout)

iplot(choromap3)
df3= pd.read_csv('../input/2014_World_Power_Consumption')

df3.head()
data3 = dict(type = 'choropleth',

           locations = df3['Country'],

           locationmode = 'country names',

           z=df3['Power Consumption KWH'],

           text = df3['Country'],

           colorbar = {'title':'Power Consumption KWH'})
layout3 = dict(title='2014 Power Consumption',

             geo = dict(showframe= False,

                       projection = {'type': 'mercator'}))
choromap3 = go.Figure(data=[data3],layout=layout3)

iplot(choromap3,validate = False)
data3 = dict(type = 'choropleth',

           locations = df3['Country'],

           colorscale = 'Viridis',

           reversescale = True,

           locationmode = 'country names',

           z=df3['Power Consumption KWH'],

           text = df3['Country'],

           colorbar = {'title':'Power Consumption KWH'})
layout3 = dict(title='2014 Power Consumption',

             geo = dict(showframe= False,

                       projection = {'type': 'mercator'}))
choromap3 = go.Figure(data=[data3],layout=layout3)

iplot(choromap3,validate = False)
df4= pd.read_csv('../input/2012_Election_Data')

df4.head()
data4 = dict(type = 'choropleth',

           colorscale = 'Viridis',

           reversescale = True,

           locations = df4['State Abv'],

           z=df4['Voting-Age Population (VAP)'],                  

           locationmode = 'USA-states',           

           text = df4['State'],

           colorbar = {'title':'Voting Age Population'})
layout4 = dict(title='2012 Election Data',

              geo = dict(scope='usa',showlakes=True,lakecolor='rgb(85,173,240)'))
choromap4 = go.Figure(data=[data4],layout=layout4)

iplot(choromap4,validate = False)