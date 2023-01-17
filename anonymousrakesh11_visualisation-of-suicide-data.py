# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools



import plotly.tools as tls

import squarify

from mpl_toolkits.basemap import Basemap



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data = pd.read_csv("../input/master.csv")

data.info()





# Any results you write to the current directory are saved as output.
data.head()
data = data.fillna(0)

data.head()
data['country'].value_counts()
data = data.drop(['country-year','HDI for year',' gdp_for_year ($) '],axis = 1)

data.head()
plt.figure(figsize=(20,10))

p= sns.barplot(x="country",y = 'suicides/100k pop', data=data, palette='colorblind')

p.set_title("variation  of suicide per 100k population with countries")

for i in p.get_xticklabels():

    i.set_rotation(90)
plt.figure(figsize=(12,6))

gender= sns.barplot(x="sex",y = 'suicides/100k pop', data=data, palette='colorblind')

gender.set_title("suicide rate per 100k population of male and female")
plt.figure(figsize=(12,6))

age1= sns.barplot(x="age",y = 'suicides/100k pop', data=data, palette='colorblind')

age1.set_title("variation of suicides par 100k population with ages ")
top_ten=pd.DataFrame(data.groupby(['country'])['suicides/100k pop'].sum().reset_index())

top_ten.sort_values(by=['suicides/100k pop'],ascending=False,inplace=True)

plt.figure(figsize = (13,7))

top10=sns.barplot(x=top_ten["country"].head(10), y=top_ten['suicides/100k pop'].head(10), data=top_ten,palette='colorblind')

top10.set_xlabel("Name of Country",fontsize=12)

top10.set_ylabel("suicides per 100k population",fontsize=12)

top10.set_title('Top 10 Countries having highest suicides')

for i in top10.get_xticklabels():

    i.set_rotation(90)
map_on=pd.DataFrame(data.groupby('country')['suicides/100k pop'].sum().reset_index())



count = [ dict(

        type = 'choropleth',

        locations = map_on['country'],

        locationmode='country names',

        z = map_on['suicides/100k pop'],

        text = map_on['country'],

        colorscale = 'earth',

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick =False,

            title = 'color based scale'),

      ) ]

layout = dict(

    title = 'Suicides across the world',

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)

fig = dict( data=count, layout=layout )

iplot( fig, validate=False, filename='d3-world-map' )