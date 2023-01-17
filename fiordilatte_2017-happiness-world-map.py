

import plotly.plotly as py  # plot package

import plotly

import numpy as np # linear algebra

import pandas as pd  # data-table package built upon numpy

import pycountry # lookup package for ISO-codes for countries of this blue, blue planet

from IPython.display import YouTubeVideo  # for displaying youtube videos

from plotly.offline import iplot, init_notebook_mode  #iplot is for plotting into a jupyter

init_notebook_mode() #turn on notebookmode

YouTubeVideo('2V-7_YxKW_k')
df = pd.read_csv('../input/2017.csv')

df = df[['Country', 'Happiness.Rank', 'Happiness.Score']]
countries= df['Country'].values



def lookup(countries):

    result = []

    for i in range(len(countries)):

        try:

            result.append(pycountry.countries.get(name=countries[i]).alpha_3)

        except KeyError:

            try:

                result.append(pycountry.countries.get(official_name=countries[i]).alpha_3)

            except KeyError:

                result.append('undefined')

    return result
codes=lookup(countries)

df['Codes']=codes

df=df[~df.Codes.isin(['undefined'])]

data = [ dict(

    type = 'choropleth',

    locations = df['Codes'],

    z = df['Happiness.Score'],

    text = df['Country'],

   colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \

                 [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

   autocolorscale = False,

   reversescale = True,

    marker = dict(

       line = dict (

           color = 'rgb(180,180,180)',

           width = 0.5

       ) ),

    colorbar = dict(

       autotick = False,

       title = 'Happiness Score'),

) ]





layout = dict(

    title = 'Happiness Scores',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)

fig = dict( data=data, layout=layout )

iplot(fig, validate=False)
